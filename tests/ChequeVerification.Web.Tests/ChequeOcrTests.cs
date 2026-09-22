using System.Text.Json;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Controllers;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

public class ChequeOcrTests : IDisposable
{
    private readonly string _webRoot;
    private readonly FakeOcrApiClient _api = new();

    public ChequeOcrTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "verify-ocr-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_webRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { }
    }

    [Fact]
    public void Dto_Deserializes_ValidOcrResponse()
    {
        const string json = """
            {
              "success": true,
              "message": "OCR effectuée avec succès.",
              "full_text": "BANQUE ALGERIE\nPAYEZ CONTRE CE CHEQUE",
              "lines": [
                {"text": "BANQUE ALGERIE", "confidence": 0.98, "box": [[0,0],[10,0],[10,10],[0,10]]},
                {"text": "PAYEZ CONTRE CE CHEQUE", "confidence": 0.95, "box": [[0,12],[20,12],[20,22],[0,22]]}
              ],
              "fields": {
                "cheque_number": "123456",
                "date": "12/08/2026",
                "amount_text": "1000 DA",
                "amount_numeric": "1000.00",
                "account_number": "123456789"
              },
              "processing_ms": 120,
              "lang": "fr",
              "device": "cpu"
            }
            """;
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web);
        var dto = JsonSerializer.Deserialize<ChequeOcrResponseDto>(json, options);
        Assert.NotNull(dto);
        Assert.True(dto!.Success);
        Assert.Equal("BANQUE ALGERIE\nPAYEZ CONTRE CE CHEQUE", dto.FullText);
        Assert.Equal(2, dto.Lines.Count);
        Assert.Equal(0.98, dto.Lines[0].Confidence, precision: 3);
        Assert.Equal("123456", dto.Fields.ChequeNumber);
        Assert.Equal("fr", dto.Lang);
        Assert.Equal("cpu", dto.Device);
        Assert.Equal(120, dto.ProcessingMs);
    }

    [Fact]
    public async Task Service_503_HandledGracefully()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        _api.OcrResult = new ChequeOcrResponseDto { Success = false, Message = "Service OCR indisponible.", FullText = "", Lines = new(), Fields = new(), ProcessingMs = 0, Lang = "fr", Device = "unavailable" };

        var service = CreateVerificationService(db);
        var result = await service.OcrChequeAsync(1);

        Assert.False(result.Success);
        Assert.Contains("indisponible", result.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("unavailable", result.Device);
    }

    [Fact]
    public async Task Service_UnavailableApi_HandledGracefully()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        _api.OcrResult = null; // unreachable

        var service = CreateVerificationService(db);
        var result = await service.OcrChequeAsync(1);

        Assert.False(result.Success);
        Assert.Contains("indisponible", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ArbitraryFilePath_CannotBeSupplied_ControllerOnlyAcceptsChequeId()
    {
        // ChequeOcrOperationResult is requested via ChequeId only; path is resolved server-side.
        var method = typeof(VerificationsController).GetMethod("TestChequeOcr");
        Assert.NotNull(method);
        var parms = method!.GetParameters();
        // First param must be chequeId int; second is CancellationToken; no string path
        Assert.True(parms.Length >= 1);
        Assert.Equal("chequeId", parms[0].Name);
        Assert.Equal(typeof(int), parms[0].ParameterType);
        Assert.DoesNotContain(parms, p => p.ParameterType == typeof(string) && p.Name!.ToLower().Contains("path"));

        // Service interface also only takes ChequeId
        var svcMethod = typeof(IVerificationService).GetMethod("OcrChequeAsync");
        Assert.NotNull(svcMethod);
        Assert.Equal(typeof(int), svcMethod!.GetParameters()[0].ParameterType);

        // End-to-end: service resolves physical path from DB, not from caller
        await using var db = CreateDbContext();
        await SeedAsync(db);
        _api.OcrResult = SuccessDto();
        var svc = CreateVerificationService(db);
        var ok = await svc.OcrChequeAsync(1);
        Assert.True(ok.Success);
        // No way to inject "../../../etc/passwd"
        Assert.DoesNotContain("..", ok.FullText);
    }

    [Fact]
    public async Task NoDbPersistence_OnOcr()
    {
        await using var db = new CountingSaveChangesDbContext(CreateOptions());
        await SeedAsync(db);
        _api.OcrResult = SuccessDto();
        db.ResetSaveChangesCount();

        var service = new VerificationService(db, _api, new FakeWebHostEnvironment { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance);
        var result = await service.OcrChequeAsync(1);

        Assert.True(result.Success);
        Assert.Equal(0, db.SaveChangesCount);
        Assert.False(await db.SignatureComparisons.AnyAsync());
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.False(await db.AuditLogs.AnyAsync());
        Assert.Equal(1, (await db.Cheques.SingleAsync(c => c.ChequeId == 1)).Status);
    }

    [Fact]
    public void ControllerPost_HasAntiforgeryAndAuthorization()
    {
        var type = typeof(VerificationsController);
        var action = type.GetMethod("TestChequeOcr");
        Assert.NotNull(action);
        Assert.NotNull(action!.GetCustomAttributes(typeof(HttpPostAttribute), inherit: false).FirstOrDefault());
        Assert.NotNull(action.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), inherit: false).FirstOrDefault());
        Assert.NotNull(action.GetCustomAttributes(typeof(AuthorizeAttribute), inherit: false).FirstOrDefault());
        var auth = action.GetCustomAttributes(typeof(AuthorizeAttribute), inherit: false).Cast<AuthorizeAttribute>().First();
        Assert.Contains("Utilisateur", auth.Roles);
        Assert.Contains("Administrateur", auth.Roles);
    }

    [Fact]
    public async Task ExistingAiFlow_Unchanged_AfterOcrAddition()
    {
        // AI comparison still works identically after OCR code added
        await using var db = CreateDbContext();
        await SeedAsync(db, withExtracted: true);
        _api.OnCompareAi = _ => new SignatureAiComparisonResponseDto { Success = true, SimilarityScore = 0.75, Method = "ai_metric", Version = "v2", Model = "siamese_resnet18", Device = "cpu", Message = "ok" };
        var service = CreateVerificationService(db);
        var aiResult = await service.CompareAiSignaturesWithReferencesAsync(1);
        Assert.True(aiResult.Success);
        Assert.NotNull(aiResult.MeanRawScore);
        // OCR not called
        Assert.Equal(0, _api.OcrCallCount);
        Assert.Equal(2, _api.CompareAiCallCount);
    }

    private static ChequeOcrResponseDto SuccessDto() => new()
    {
        Success = true,
        Message = "OCR effectuée avec succès.",
        FullText = "BANQUE ALGERIE",
        Lines = new List<ChequeOcrLineDto> { new() { Text = "BANQUE ALGERIE", Confidence = 0.98, Box = new() } },
        Fields = new ChequeOcrFieldsDto { ChequeNumber = "123456" },
        ProcessingMs = 42,
        Lang = "fr",
        Device = "cpu"
    };

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>().UseInMemoryDatabase($"verify-ocr-db-{Guid.NewGuid():N}").Options;
    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());
    private VerificationService CreateVerificationService(ChequeVerificationDbContext db)
    {
        var env = new FakeWebHostEnvironment { WebRootPath = _webRoot };
        return new VerificationService(db, _api, env, NullLogger<VerificationService>.Instance);
    }

    private async Task SeedAsync(ChequeVerificationDbContext db, bool withExtracted = false)
    {
        var customer = new Customer { CustomerNumber = "CUST-OCR-1", FullName = "Client OCR", AccountNumber = "ACC-1" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        var cid = customer.CustomerId;
        // Need a real cheque image file
        var chequePath = Path.Combine(_webRoot, "cheque1.png");
        await File.WriteAllBytesAsync(chequePath, Convert.FromBase64String("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="));
        db.Cheques.Add(new Cheque { ChequeId = 1, CustomerId = cid, ImportedByUserId = 1, ChequeNumber = "CHQ-OCR-001", ImagePath = chequePath, Status = 1, UploadedAt = DateTime.UtcNow });
        if (withExtracted)
        {
            var extPath = Path.Combine(_webRoot, "extracted.png");
            await File.WriteAllBytesAsync(extPath, new byte[] { 1, 2, 3 });
            db.ExtractedSignatures.Add(new ExtractedSignature { ChequeId = 1, ImagePath = extPath, FileHash = "hash", ExtractionConfidence = 0.5m, ExtractedAt = DateTime.UtcNow });
            for (int i = 1; i <= 2; i++)
            {
                var rp = Path.Combine(_webRoot, $"ref{i}.png");
                await File.WriteAllBytesAsync(rp, new byte[] { 1, 2 });
                db.ReferenceSignatures.Add(new ReferenceSignature { ReferenceSignatureId = i, CustomerId = cid, ImagePath = rp, CreatedAt = DateTime.UtcNow, IsActive = true });
            }
        }
        await db.SaveChangesAsync();
    }

    private sealed class FakeOcrApiClient : IVerificationApiClient
    {
        public ChequeOcrResponseDto? OcrResult { get; set; }
        public int OcrCallCount { get; private set; }
        public int CompareAiCallCount { get; private set; }
        public Func<int, SignatureAiComparisonResponseDto?>? OnCompareAi { get; set; }
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default) => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<ImageAnalysisResponseDto?>(null);
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureExtractionResponseDto?>(null);
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureDebugResponseDto?>(null);
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default) => Task.FromResult<SignatureComparisonResponseDto?>(null);
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream es, string ef, string ec, Stream rs, string rf, string rc, CancellationToken ct = default)
        {
            CompareAiCallCount++;
            var name = Path.GetFileNameWithoutExtension(rf);
            var id = int.TryParse(name.Replace("ref", ""), out var p) ? p : CompareAiCallCount;
            return Task.FromResult(OnCompareAi?.Invoke(id));
        }
        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream s, string f, string c, CancellationToken ct = default)
        {
            OcrCallCount++;
            return Task.FromResult(OcrResult);
        }
    }

    private sealed class FakeWebHostEnvironment : IWebHostEnvironment
    {
        public string ApplicationName { get; set; } = "ChequeVerification.Web";
        public string EnvironmentName { get; set; } = "Development";
        public string ContentRootPath { get; set; } = ".";
        public string WebRootPath { get; set; } = ".";
        public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
        public IFileProvider WebRootFileProvider { get; set; } = new NullFileProvider();
    }

    private sealed class CountingSaveChangesDbContext : ChequeVerificationDbContext
    {
        public CountingSaveChangesDbContext(DbContextOptions<ChequeVerificationDbContext> o) : base(o) { }
        public int SaveChangesCount { get; private set; }
        public void ResetSaveChangesCount() => SaveChangesCount = 0;
        public override int SaveChanges(bool a = true) { SaveChangesCount++; return base.SaveChanges(a); }
        public override Task<int> SaveChangesAsync(CancellationToken ct = default) { SaveChangesCount++; return base.SaveChangesAsync(ct); }
    }
}
