using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

/// <summary>
/// Bilan extraction section: persisted identity comes from the
/// ExtractedSignatures row (no API); pipeline evidence comes from the
/// existing live V2.4 debug endpoint (read-only, best-effort).
/// </summary>
public class ExtractionBilanTests : IDisposable
{
    private readonly string _webRoot;
    private readonly FakeDebugApiClient _api = new();

    public ExtractionBilanTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "verify-ext-bilan-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_webRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { }
    }

    [Fact]
    public async Task DetailsMapping_ExposesPersistedExtractionIdentity_WithoutApiCalls()
    {
        await using var db = CreateDbContext();
        var extractedAt = new DateTime(2026, 5, 1, 10, 0, 0, DateTimeKind.Utc);
        var verificationId = await SeedDetailsAsync(db, extractedAt);
        var svc = new VerificationService(db, new ThrowingDebugApiClient(),
            new FakeEnv { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance);

        var details = await svc.GetVerificationDetailsAsync(verificationId);

        Assert.NotNull(details);
        Assert.Equal(0.785m, details!.ExtractionQuality);
        Assert.NotNull(details.ExtractedSignatureId);
        Assert.Equal(extractedAt, details.ExtractedAt);
        Assert.Null(details.ExtractionDiagnostic);
    }

    [Fact]
    public async Task ExtractionDiagnostic_MapsRealV24Evidence()
    {
        await using var db = CreateDbContext();
        var chequePath = Path.Combine(_webRoot, "cheque.png");
        await File.WriteAllBytesAsync(chequePath, new byte[] { 1, 2, 3 });
        var customer = new Customer { CustomerNumber = "CUST-EXT", FullName = "Ext", AccountNumber = "ACC" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        db.Cheques.Add(new Cheque
        {
            ChequeId = 1, CustomerId = customer.CustomerId, ImportedByUserId = 1,
            ChequeNumber = "CHQ-EXT", ImagePath = chequePath, Status = 1, UploadedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();
        _api.OnDebug = () => new SignatureDebugResponseDto
        {
            Success = true,
            ExtractionPipelineVersion = "2.4",
            OriginalWidth = 800,
            OriginalHeight = 355,
            CandidateRoi = new BoundingBoxDto { X = 400, Y = 195, Width = 384, Height = 142 },
            MicrBand = new BoundingBoxDto { X = 0, Y = 100, Width = 384, Height = 42 },
            SignatureBbox = new BoundingBoxDto { X = 128, Y = 30, Width = 185, Height = 91 },
            TotalComponentCount = 12,
            RetainedComponentCount = 7,
            RejectedComponentCount = 5,
            MicrRejectedCount = 2,
            GroupCount = 3,
            SelectedGroupIndex = 1,
            SelectionReason = "score_max",
            RejectedComponentReasons = new Dictionary<string, int> { ["aire"] = 3 },
            DominantComponentBbox = new BoundingBoxDto { X = 10, Y = 5, Width = 60, Height = 40 },
            DominantComponentInk = 900,
            DominantComponentScore = 0.42,
            RefinedComponentCount = 5,
            CoreComponentIndices = new List<int> { 0, 1, 2 },
            RefinedSignatureBbox = new BoundingBoxDto { X = 120, Y = 28, Width = 190, Height = 95 },
            RefinementReason = "core_connected",
            DirectionalAcceptedIndices = new List<int> { 3 },
            CompletenessScore = 0.91,
            CompletenessRefinedInk = 1200,
            CompletenessReferenceInk = 1318,
            QualityBase = 0.82,
            QualityCompletenessFactor = 0.957,
            ExtractionQuality = 0.785,
            OriginalWithRoiBase64 = "AAA=",
            RoiImageBase64 = "BBB=",
            MaskImageBase64 = "CCC=",
            RefinedGroupImageBase64 = "DDD="
        };
        var svc = CreateService(db);

        var diag = await svc.GetExtractionDiagnosticAsync(1);

        Assert.True(diag.Success);
        Assert.Equal(384, diag.RoiWidth);
        Assert.Equal(195, diag.RoiY);
        Assert.Equal(185, diag.BboxWidth);
        Assert.Equal(12, diag.TotalComponentCount);
        Assert.Equal(2, diag.MicrRejectedCount);
        Assert.Equal(3, diag.GroupCount);
        Assert.Equal(1, diag.SelectedGroupIndex);
        Assert.Equal(new List<int> { 3 }, diag.DirectionalAcceptedIndices);
        Assert.Equal(0.91, diag.CompletenessScore);
        Assert.Equal(0.82, diag.QualityBase);
        Assert.StartsWith("data:image/png;base64,", diag.OriginalWithRoiDataUri);
        Assert.StartsWith("data:image/png;base64,", diag.RoiImageDataUri);
        Assert.StartsWith("data:image/png;base64,", diag.MaskImageDataUri);
        Assert.StartsWith("data:image/png;base64,", diag.RefinedGroupImageDataUri);
        Assert.True(string.IsNullOrEmpty(diag.SignatureImageDataUri));
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.False(await db.ExtractedSignatures.AnyAsync());
    }

    [Fact]
    public async Task ExtractionDiagnostic_ApiUnavailable_ReturnsFailureWithoutThrow()
    {
        await using var db = CreateDbContext();
        var chequePath = Path.Combine(_webRoot, "cheque.png");
        await File.WriteAllBytesAsync(chequePath, new byte[] { 1 });
        var customer = new Customer { CustomerNumber = "CUST-EXT2", FullName = "Ext2", AccountNumber = "ACC" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        db.Cheques.Add(new Cheque
        {
            ChequeId = 1, CustomerId = customer.CustomerId, ImportedByUserId = 1,
            ChequeNumber = "CHQ-EXT2", ImagePath = chequePath, Status = 1, UploadedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();
        _api.OnDebug = null;
        var svc = CreateService(db);

        var diag = await svc.GetExtractionDiagnosticAsync(1);

        Assert.False(diag.Success);
        Assert.False(string.IsNullOrWhiteSpace(diag.ErrorMessage));
    }

    private async Task<int> SeedDetailsAsync(ChequeVerificationDbContext db, DateTime extractedAt)
    {
        var customer = new Customer { CustomerNumber = "CUST-BIL", FullName = "Bilan", AccountNumber = "ACC" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        var cheque = new Cheque
        {
            CustomerId = customer.CustomerId, ImportedByUserId = 1, ChequeNumber = "CHQ-BIL",
            ImagePath = "/img.png", Status = 3, UploadedAt = DateTime.UtcNow
        };
        db.Cheques.Add(cheque);
        await db.SaveChangesAsync();
        var extracted = new ExtractedSignature
        {
            ChequeId = cheque.ChequeId, ImagePath = "/ext.png", FileHash = "h",
            ExtractionConfidence = 0.785m, ExtractedAt = extractedAt
        };
        db.ExtractedSignatures.Add(extracted);
        var reference = new ReferenceSignature
        {
            CustomerId = customer.CustomerId, ImagePath = "/ref.png",
            CreatedAt = DateTime.UtcNow, IsActive = true
        };
        db.ReferenceSignatures.Add(reference);
        await db.SaveChangesAsync();
        var vr = new VerificationResult
        {
            ChequeId = cheque.ChequeId, SimilarityScore = 0.95m,
            LowerThresholdUsed = 0.6585m, UpperThresholdUsed = 0.9150m,
            AutomaticDecision = 1, FinalDecision = 1,
            ModelName = "sig-verif-ai-v5a", ModelVersion = "v5a-phase7",
            VerifiedAt = DateTime.UtcNow
        };
        db.VerificationResults.Add(vr);
        await db.SaveChangesAsync();
        db.SignatureComparisons.Add(new SignatureComparison
        {
            VerificationId = vr.VerificationId, ExtractedSignatureId = extracted.ExtractedSignatureId,
            ReferenceSignatureId = reference.ReferenceSignatureId, SimilarityScore = 0.95m, IsBestMatch = true
        });
        await db.SaveChangesAsync();
        return vr.VerificationId;
    }

    private ChequeVerificationDbContext CreateDbContext()
        => new(new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"verify-ext-bilan-{Guid.NewGuid():N}").Options);
    private VerificationService CreateService(ChequeVerificationDbContext db)
        => new(db, _api, new FakeEnv { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance);

    private sealed class FakeDebugApiClient : IVerificationApiClient
    {
        public Func<SignatureDebugResponseDto?>? OnDebug { get; set; }
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default)
            => Task.FromResult<HealthResponseDto?>(null);
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult<ImageAnalysisResponseDto?>(null);
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult<SignatureExtractionResponseDto?>(null);
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult(OnDebug?.Invoke());
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default)
            => Task.FromResult<SignatureComparisonResponseDto?>(null);
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default)
            => Task.FromResult<SignatureAiComparisonResponseDto?>(null);
        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult<ChequeOcrResponseDto?>(null);
    }

    private sealed class ThrowingDebugApiClient : IVerificationApiClient
    {
        private static T Throw<T>() => throw new InvalidOperationException("No API call expected on read.");
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default) => Throw<Task<HealthResponseDto?>>();
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string f, string c, CancellationToken ct = default) => Throw<Task<ImageAnalysisResponseDto?>>();
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string f, string c, CancellationToken ct = default) => Throw<Task<SignatureExtractionResponseDto?>>();
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string f, string c, CancellationToken ct = default) => Throw<Task<SignatureDebugResponseDto?>>();
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default) => Throw<Task<SignatureComparisonResponseDto?>>();
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default) => Throw<Task<SignatureAiComparisonResponseDto?>>();
        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream s, string f, string c, CancellationToken ct = default) => Throw<Task<ChequeOcrResponseDto?>>();
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
