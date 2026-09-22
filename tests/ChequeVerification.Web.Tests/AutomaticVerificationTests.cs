using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace ChequeVerification.Web.Tests;

/// <summary>
/// Automatic User flow: one call runs V2.4 extraction then V5-A K=5 decision.
/// Algorithms, thresholds and policy are reused, never rebuilt here.
/// </summary>
public class AutomaticVerificationTests : IDisposable
{
    private readonly string _webRoot;
    private readonly FakeAutoApiClient _api = new();

    private const string TinyPngBase64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    public AutomaticVerificationTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "verify-auto-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_webRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { }
    }

    [Fact]
    public async Task PendingCheque_ManualZone_PersistsBilanWithNullFinalDecision()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, chequeStatus: ChequeStatus.EnAttente, extracted: false);
        _api.OnExtract = ValidExtraction;
        _api.OnCompareAi = _ => AiSuccess(0.75);
        var svc = CreateService(db);

        var res = await svc.VerifyAutomaticallyAsync(1, 1);

        Assert.True(res.Success);
        Assert.NotNull(res.VerificationId);
        var vr = await db.VerificationResults.SingleAsync(v => v.ChequeId == 1);
        Assert.Equal(VerificationDecision.ControleManuel, vr.AutomaticDecision);
        Assert.Null(vr.FinalDecision);
        Assert.Equal(0.6585m, vr.LowerThresholdUsed);
        Assert.Equal(0.9150m, vr.UpperThresholdUsed);
        Assert.Equal(ChequeStatus.ControleManuel, (await db.Cheques.SingleAsync(c => c.ChequeId == 1)).Status);
        var comps = await db.SignatureComparisons.Where(s => s.VerificationId == vr.VerificationId).ToListAsync();
        Assert.Equal(5, comps.Count);
        Assert.Single(comps, c => c.IsBestMatch);
        Assert.True(await db.ExtractedSignatures.AnyAsync(e => e.ChequeId == 1));
    }

    [Fact]
    public async Task NonPendingCheque_RejectedBeforeAnyApiCall()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, chequeStatus: ChequeStatus.Rejete, extracted: false);
        var svc = CreateService(db);

        var res = await svc.VerifyAutomaticallyAsync(1, 1);

        Assert.False(res.Success);
        Assert.Equal(0, _api.ExtractionCallCount);
        Assert.Equal(0, _api.AiComparisonCallCount);
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.False(await db.ExtractedSignatures.AnyAsync());
    }

    [Fact]
    public async Task ExtractionFailure_BlocksAiAndPersistence()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, chequeStatus: ChequeStatus.EnAttente, extracted: false);
        _api.OnExtract = null;
        var svc = CreateService(db);

        var res = await svc.VerifyAutomaticallyAsync(1, 1);

        Assert.False(res.Success);
        Assert.Equal(0, _api.AiComparisonCallCount);
        Assert.False(await db.VerificationResults.AnyAsync());
    }

    [Fact]
    public void VerifyAutomatically_RequiresUtilisateurOrAdministrateur()
    {
        var method = typeof(ChequeVerification.Web.Controllers.VerificationsController).GetMethod("VerifyAutomatically");
        Assert.NotNull(method);
        var auth = method!.GetCustomAttributes(typeof(Microsoft.AspNetCore.Authorization.AuthorizeAttribute), false)
            .Cast<Microsoft.AspNetCore.Authorization.AuthorizeAttribute>().ToList();
        Assert.NotEmpty(auth);
        var roles = string.Join(",", auth.Select(a => a.Roles));
        Assert.Contains("Utilisateur", roles);
        Assert.Contains("Administrateur", roles);
        Assert.DoesNotContain("Contrôleur", roles);
        Assert.NotNull(method.GetCustomAttributes(typeof(Microsoft.AspNetCore.Mvc.ValidateAntiForgeryTokenAttribute), false).FirstOrDefault());
        Assert.NotNull(method.GetCustomAttributes(typeof(Microsoft.AspNetCore.Mvc.HttpPostAttribute), false).FirstOrDefault());
    }

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"verify-auto-db-{Guid.NewGuid():N}").Options;
    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());
    private VerificationService CreateService(ChequeVerificationDbContext db)
        => new(db, _api, new FakeEnv { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance,
            Options.Create(new VerificationPolicyOptions { LowerThreshold = 0.6585m, UpperThreshold = 0.9150m }));

    private async Task SeedAsync(ChequeVerificationDbContext db, byte chequeStatus, bool extracted)
    {
        var cust = new Customer { CustomerNumber = "CUST-AUTO", FullName = "Auto", AccountNumber = "ACC" };
        db.Customers.Add(cust);
        await db.SaveChangesAsync();
        for (var i = 1; i <= 5; i++)
        {
            var p = Path.Combine(_webRoot, $"ref{i}.png");
            await File.WriteAllBytesAsync(p, new byte[] { 1, 2, 3 });
            db.ReferenceSignatures.Add(new ReferenceSignature
            {
                ReferenceSignatureId = i, CustomerId = cust.CustomerId, ImagePath = p,
                CreatedAt = DateTime.UtcNow.AddMinutes(-i), IsActive = true
            });
        }
        var chequePath = Path.Combine(_webRoot, "cheque.png");
        await File.WriteAllBytesAsync(chequePath, new byte[] { 7, 7, 7 });
        db.Cheques.Add(new Cheque
        {
            ChequeId = 1, CustomerId = cust.CustomerId, ImportedByUserId = 1,
            ChequeNumber = "CHQ-AUTO", ImagePath = chequePath, Status = chequeStatus, UploadedAt = DateTime.UtcNow
        });
        if (extracted)
        {
            var extPath = Path.Combine(_webRoot, "ext.png");
            await File.WriteAllBytesAsync(extPath, new byte[] { 9 });
            db.ExtractedSignatures.Add(new ExtractedSignature
            {
                ChequeId = 1, ImagePath = extPath, FileHash = "h",
                ExtractionConfidence = 0.5m, ExtractedAt = DateTime.UtcNow
            });
        }
        await db.SaveChangesAsync();
    }

    private static SignatureExtractionResponseDto ValidExtraction() => new()
    {
        Success = true, OriginalWidth = 800, OriginalHeight = 355,
        ExtractionQuality = 0.6, ImageFormat = "png", SignatureImageBase64 = TinyPngBase64
    };

    private static SignatureAiComparisonResponseDto AiSuccess(double s) => new()
    {
        Success = true, SimilarityScore = s, Method = "ai_metric",
        Version = "v2", Model = "siamese_resnet18", Device = "cpu", Message = "ok"
    };

    private sealed class FakeAutoApiClient : IVerificationApiClient
    {
        public Func<SignatureExtractionResponseDto?>? OnExtract { get; set; }
        public Func<int, SignatureAiComparisonResponseDto?>? OnCompareAi { get; set; }
        public int ExtractionCallCount { get; private set; }
        public int AiComparisonCallCount { get; private set; }
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default)
            => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult<ImageAnalysisResponseDto?>(null);
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string f, string c, CancellationToken ct = default)
        {
            ExtractionCallCount++;
            return Task.FromResult(OnExtract?.Invoke());
        }
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult<SignatureDebugResponseDto?>(null);
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default)
            => Task.FromResult<SignatureComparisonResponseDto?>(null);
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream es, string ef, string ec, Stream rs, string rf, string rc, CancellationToken ct = default)
        {
            AiComparisonCallCount++;
            var name = Path.GetFileNameWithoutExtension(rf);
            var id = int.TryParse(name.Replace("ref", string.Empty), out var p) ? p : AiComparisonCallCount;
            return Task.FromResult(OnCompareAi?.Invoke(id));
        }
        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult<ChequeOcrResponseDto?>(null);
    }

    private sealed class FakeEnv : IWebHostEnvironment
    {
        public string ApplicationName { get; set; } = "test";
        public string EnvironmentName { get; set; } = "Development";
        public string ContentRootPath { get; set; } = ".";
        public string WebRootPath { get; set; } = ".";
        public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
        public IFileProvider WebRootFileProvider { get; set; } = new NullFileProvider();
    }
}
