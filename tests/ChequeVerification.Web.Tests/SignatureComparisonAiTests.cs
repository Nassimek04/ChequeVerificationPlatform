using System.Text.Json;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Controllers;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Verifications;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

public class SignatureComparisonAiTests : IDisposable
{
    private readonly string _webRoot;
    private readonly FakeAiVerificationApiClient _api = new();

    public SignatureComparisonAiTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "verify-compare-ai-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_webRoot);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_webRoot))
            {
                Directory.Delete(_webRoot, recursive: true);
            }
        }
        catch
        {
            // Best-effort cleanup only.
        }
    }

    [Fact]
    public async Task NoExtractedSignature_BlocksOperation()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, createExtracted: false);
        _api.OnCompareAi = _ => SuccessDto(0.9);

        var service = CreateService(db);
        var result = await service.CompareAiSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.Contains("Aucune signature n'a été extraite", result.Message);
        Assert.Empty(result.Comparisons);
        Assert.Equal(0, _api.CompareAiCallCount);
    }

    [Fact]
    public async Task NoActiveReference_BlocksOperation()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 0);
        _api.OnCompareAi = _ => SuccessDto(0.9);

        var service = CreateService(db);
        var result = await service.CompareAiSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.Contains("Aucune signature de référence active", result.Message);
        Assert.Empty(result.Comparisons);
        Assert.Equal(0, _api.CompareAiCallCount);
    }

    [Fact]
    public async Task ExtractedFileMissing_BlocksOperation()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        File.Delete(_webRoot + "\\extracted.png");
        _api.OnCompareAi = _ => SuccessDto(0.9);

        var service = CreateService(db);
        var result = await service.CompareAiSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.Contains("introuvable", result.Message);
        Assert.Equal(0, _api.CompareAiCallCount);
    }

    [Fact]
    public async Task MultipleReferences_AllComparedWithIndependentScores()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 3);
        _api.OnCompareAi = referenceId => SuccessDto(referenceId == 2 ? 0.838154 : 0.4);

        var service = CreateService(db);
        var result = await service.CompareAiSignaturesWithReferencesAsync(1);

        Assert.True(result.Success, result.Message);
        Assert.Contains("3/3", result.Message);
        Assert.Equal(3, result.Comparisons.Count);
        Assert.All(result.Comparisons, c =>
        {
            Assert.True(c.IsAvailable);
            Assert.NotNull(c.Score);
            Assert.InRange(c.Score!.Value, -1.0, 1.0);
            Assert.Equal("ai_metric", c.Method);
            Assert.Equal("v2", c.Version);
            Assert.Equal("siamese_resnet18", c.Model);
        });
        // The OpenCV baseline is NOT called by the AI flow (independence).
        Assert.Equal(0, _api.CompareCallCount);
        Assert.Equal(3, _api.CompareAiCallCount);
    }

    [Fact]
    public async Task Structured503Unavailable_IsSurfacedAsStatusMessage_NotFailureCrash()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 2);
        _api.OnCompareAi = _ => new SignatureAiComparisonResponseDto
        {
            Success = false,
            SimilarityScore = null,
            Method = "ai_metric",
            Version = "v2",
            Model = "siamese_resnet18",
            EmbeddingDimension = 128,
            Device = "unavailable",
            Message = "Modèle IA non chargé."
        };

        var service = CreateService(db);
        var result = await service.CompareAiSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.Contains("indisponible", result.Message);
        Assert.Equal(2, result.Comparisons.Count);
        Assert.All(result.Comparisons, c =>
        {
            Assert.False(c.IsAvailable);
            Assert.Null(c.Score);
            Assert.Contains("Comparaison IA indisponible", c.StatusMessage);
        });
    }

    [Fact]
    public async Task ApiDown_NoCrash_ComparisonsMarkedUnavailable()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 2);
        _api.OnCompareAi = _ => null;

        var service = CreateService(db);
        var result = await service.CompareAiSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.Contains("indisponible", result.Message);
        Assert.All(result.Comparisons, c =>
        {
            Assert.False(c.IsAvailable);
            Assert.Null(c.Score);
        });
    }

    [Fact]
    public async Task AiComparison_IsReadOnly_NoSqlChangeNoAuditNoStatusChange_NoPersistence()
    {
        await using var db = new CountingSaveChangesDbContext(CreateOptions());
        await SeedAsync(db, references: 2);
        _api.OnCompareAi = _ => SuccessDto(0.7);

        db.ResetSaveChangesCount();

        var service = new VerificationService(
            db,
            _api,
            new FakeWebHostEnvironment { WebRootPath = _webRoot },
            NullLogger<VerificationService>.Instance);
        var result = await service.CompareAiSignaturesWithReferencesAsync(1);

        Assert.True(result.Success);

        // Zero SQL writes, zero AI persistence.
        Assert.Equal(0, db.SaveChangesCount);
        Assert.Single(await db.ExtractedSignatures.ToListAsync());
        Assert.Equal(2, await db.ReferenceSignatures.CountAsync());
        Assert.False(await db.AuditLogs.AnyAsync());
        Assert.False(await db.SignatureComparisons.AnyAsync());
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.Equal(1, (await db.Cheques.SingleAsync(c => c.ChequeId == 1)).Status);
    }

    [Fact]
    public async Task TestSignatureComparisonAiAction_MapsResultIntoViewModel_ZeroSqlWrites()
    {
        await using var db = new CountingSaveChangesDbContext(CreateOptions());
        await SeedAsync(db, references: 3);
        _api.OnCompareAi = referenceId => SuccessDto(referenceId == 1 ? 0.91 : 0.3);

        db.ResetSaveChangesCount();

        var service = CreateService(db);
        var controller = new VerificationsController(
            service,
            _api,
            new FakeWebHostEnvironment { WebRootPath = _webRoot });

        var actionResult = await controller.TestSignatureComparisonAi(1);
        var viewResult = Assert.IsType<ViewResult>(actionResult);
        var model = Assert.IsType<VerificationCreateViewModel>(viewResult.Model);

        Assert.NotNull(model.SignatureAiComparison);
        Assert.True(model.SignatureAiComparison!.Success);
        Assert.Equal(3, model.SignatureAiComparison.Comparisons.Count);
        Assert.True(model.SignatureAiComparison.Comparisons.Single(c => c.ReferenceSignatureId == 1).Score > 0.9);

        Assert.Equal(0, db.SaveChangesCount);
        Assert.False(await db.SignatureComparisons.AnyAsync());
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.False(await db.AuditLogs.AnyAsync());
    }

    [Fact]
    public void Dto_SnakeCaseProperties_Deserialized_IncludingNullScoreAndBboxes()
    {
        const string json = """
            {
              "success": true,
              "similarity_score": 0.838154,
              "method": "ai_metric",
              "version": "v2",
              "model": "siamese_resnet18",
              "embedding_dimension": 128,
              "device": "cuda",
              "message": "Comparaison IA effectuée.",
              "extracted_ink_bbox": { "x": 10, "y": 20, "width": 30, "height": 40 },
              "reference_ink_bbox": null
            }
            """;

        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web);
        var dto = JsonSerializer.Deserialize<SignatureAiComparisonResponseDto>(json, options);

        Assert.NotNull(dto);
        Assert.True(dto!.Success);
        Assert.Equal(0.838154, dto.SimilarityScore);
        Assert.Equal("ai_metric", dto.Method);
        Assert.Equal("v2", dto.Version);
        Assert.Equal("siamese_resnet18", dto.Model);
        Assert.Equal(128, dto.EmbeddingDimension);
        Assert.Equal("cuda", dto.Device);
        Assert.NotNull(dto.ExtractedInkBBox);
        Assert.Equal(30, dto.ExtractedInkBBox!.Width);
        Assert.Null(dto.ReferenceInkBBox);
    }

    [Fact]
    public void Dto_UnavailableResponse_NullScoreDeserialized()
    {
        const string json = """
            {
              "success": false,
              "similarity_score": null,
              "method": "ai_metric",
              "version": "v2",
              "model": "siamese_resnet18",
              "embedding_dimension": 128,
              "device": "unavailable",
              "message": "Service IA indisponible."
            }
            """;

        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web);
        var dto = JsonSerializer.Deserialize<SignatureAiComparisonResponseDto>(json, options);

        Assert.NotNull(dto);
        Assert.False(dto!.Success);
        Assert.Null(dto.SimilarityScore);
        Assert.Equal("unavailable", dto.Device);
    }

    [Fact]
    public void Controller_HasAuthorizationAndActionHasAntiforgery()
    {
        var type = typeof(VerificationsController);

        var action = type.GetMethod("TestSignatureComparisonAi");
        Assert.NotNull(action);
        Assert.NotNull(action!.GetCustomAttributes(typeof(HttpPostAttribute), inherit: false).FirstOrDefault());
        Assert.NotNull(action.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), inherit: false).FirstOrDefault());
        Assert.NotNull(action.GetCustomAttributes(typeof(AuthorizeAttribute), inherit: false).FirstOrDefault());
    }

    private static SignatureAiComparisonResponseDto SuccessDto(double score) => new()
    {
        Success = true,
        SimilarityScore = score,
        Method = "ai_metric",
        Version = "v2",
        Model = "siamese_resnet18",
        EmbeddingDimension = 128,
        Device = "cpu",
        Message = "Comparaison IA effectuée."
    };

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
    {
        return new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"verify-compare-ai-db-{Guid.NewGuid():N}")
            .Options;
    }

    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());

    private VerificationService CreateService(ChequeVerificationDbContext db)
    {
        var env = new FakeWebHostEnvironment { WebRootPath = _webRoot };
        return new VerificationService(db, _api, env, NullLogger<VerificationService>.Instance);
    }

    private async Task SeedAsync(ChequeVerificationDbContext db, int references = 2, bool createExtracted = true)
    {
        var customer = new Customer
        {
            CustomerNumber = "CUST-AI-1",
            FullName = "Jean Dupont",
            AccountNumber = "ACC-AI-1"
        };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        var customerId = customer.CustomerId;

        for (var i = 1; i <= references; i++)
        {
            var refPath = Path.Combine(_webRoot, $"ref{i}.png");
            await File.WriteAllBytesAsync(refPath, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            db.ReferenceSignatures.Add(new ReferenceSignature
            {
                ReferenceSignatureId = i,
                CustomerId = customerId,
                ImagePath = refPath,
                CreatedAt = DateTime.UtcNow,
                IsActive = true
            });
        }

        db.Cheques.Add(new Cheque
        {
            ChequeId = 1,
            CustomerId = customerId,
            ImportedByUserId = 1,
            ChequeNumber = "CHQ-AI-0001",
            ImagePath = _webRoot + "\\cheque1.png",
            Status = 1,
            UploadedAt = DateTime.UtcNow
        });

        if (createExtracted)
        {
            var extractedPath = Path.Combine(_webRoot, "extracted.png");
            await File.WriteAllBytesAsync(extractedPath, new byte[] { 9, 9, 9, 9 });
            db.ExtractedSignatures.Add(new ExtractedSignature
            {
                ChequeId = 1,
                ImagePath = extractedPath,
                FileHash = "hash",
                ExtractionConfidence = 0.6m,
                ExtractedAt = DateTime.UtcNow
            });
        }

        await db.SaveChangesAsync();

        await File.WriteAllBytesAsync(_webRoot + "\\cheque1.png", new byte[] { 1, 1, 1 });
    }

    private sealed class FakeAiVerificationApiClient : IVerificationApiClient
    {
        public Func<int, SignatureAiComparisonResponseDto?>? OnCompareAi { get; set; }
        public int CompareAiCallCount { get; private set; }
        public int CompareCallCount { get; private set; }

        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken cancellationToken = default)
            => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });

        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<ImageAnalysisResponseDto?>(null);

        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<SignatureExtractionResponseDto?>(null);

        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<SignatureDebugResponseDto?>(null);

        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default)
        {
            CompareCallCount++;
            return Task.FromResult<SignatureComparisonResponseDto?>(null);
        }

        public async Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default)
        {
            CompareAiCallCount++;
            // Deterministic: derive the reference id from the uploaded file name
            // ("refN.png") instead of relying on enumeration order.
            var name = Path.GetFileNameWithoutExtension(referenceFileName);
            var referenceId = int.TryParse(name.Replace("ref", string.Empty), out var parsed) ? parsed : CompareAiCallCount;
            return await Task.FromResult(OnCompareAi?.Invoke(referenceId));
        }

        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream a, string b, string c, CancellationToken ct = default) => Task.FromResult<ChequeOcrResponseDto?>(null);
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
        public CountingSaveChangesDbContext(DbContextOptions<ChequeVerificationDbContext> options)
            : base(options)
        {
        }

        public int SaveChangesCount { get; private set; }

        public void ResetSaveChangesCount() => SaveChangesCount = 0;

        public override int SaveChanges(bool acceptAllChangesOnSuccess = true)
        {
            SaveChangesCount++;
            return base.SaveChanges(acceptAllChangesOnSuccess);
        }

        public override Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
        {
            SaveChangesCount++;
            return base.SaveChangesAsync(cancellationToken);
        }
    }
}
