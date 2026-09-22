using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.FileProviders;

namespace ChequeVerification.Web.Tests;

public class AiMultiReferenceAggregationTests : IDisposable
{
    private readonly string _webRoot;
    private readonly FakeAiVerificationApiClient _api = new();

    public AiMultiReferenceAggregationTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "verify-agg-ai-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_webRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { }
    }

    [Fact]
    public async Task K1_MeanEqualsScore()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 1);
        _api.OnCompareAi = _ => SuccessDto(0.5);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.True(result.Success);
        Assert.Equal(1, result.ActiveReferenceCount);
        Assert.Equal(1, result.ComparedReferenceCount);
        Assert.Equal(0, result.UnavailableReferenceCount);
        Assert.True(result.IsAggregationAvailable);
        Assert.NotNull(result.MeanRawScore);
        Assert.Equal(0.5, result.MeanRawScore.Value, precision: 6);
    }

    [Fact]
    public async Task K2_ArithmeticMeanCorrect()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 2);
        _api.OnCompareAi = id => SuccessDto(id == 1 ? 0.2 : 0.4);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.True(result.Success);
        Assert.Equal(2, result.ActiveReferenceCount);
        Assert.Equal(2, result.ComparedReferenceCount);
        Assert.Equal(0, result.UnavailableReferenceCount);
        Assert.Equal(0.3, result.MeanRawScore!.Value, precision: 6);
    }

    [Fact]
    public async Task K3_ArithmeticMeanCorrect()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 3);
        _api.OnCompareAi = id => SuccessDto(id == 1 ? 0.2 : id == 2 ? 0.4 : 0.6);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.Equal(0.4, result.MeanRawScore!.Value, precision: 6);
        Assert.Equal(3, result.ActiveReferenceCount);
        Assert.Equal(3, result.ComparedReferenceCount);
    }

    [Fact]
    public async Task K4_ArithmeticMeanCorrect()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 4);
        _api.OnCompareAi = id => SuccessDto(id * 0.1 + 0.1); // 0.2,0.3,0.4,0.5

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        var expected = (0.2 + 0.3 + 0.4 + 0.5) / 4.0;
        Assert.Equal(expected, result.MeanRawScore!.Value, precision: 6);
        Assert.Equal(4, result.ActiveReferenceCount);
    }

    [Fact]
    public async Task K5_ArithmeticMeanCorrect()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 5);
        _api.OnCompareAi = id => SuccessDto(id == 1 ? 0.1 : id == 2 ? 0.3 : id == 3 ? 0.5 : id == 4 ? 0.7 : 0.9);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        var expected = (0.1 + 0.3 + 0.5 + 0.7 + 0.9) / 5.0;
        Assert.Equal(expected, result.MeanRawScore!.Value, precision: 6);
        Assert.Equal(5, result.ActiveReferenceCount);
        Assert.Equal(5, result.ComparedReferenceCount);
        Assert.Equal(0, result.UnavailableReferenceCount);
        Assert.True(result.IsAggregationAvailable);
    }

    [Fact]
    public async Task K5_OneUnavailable_MeanOver4()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 5);
        // id 3 unavailable via API null
        _api.OnCompareAi = id => id == 3 ? null : SuccessDto(id == 1 ? 0.2 : id == 2 ? 0.4 : id == 4 ? 0.6 : 0.8);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.True(result.Success);
        Assert.Equal(5, result.ActiveReferenceCount);
        Assert.Equal(4, result.ComparedReferenceCount);
        Assert.Equal(1, result.UnavailableReferenceCount);
        var expected = (0.2 + 0.4 + 0.6 + 0.8) / 4.0;
        Assert.Equal(expected, result.MeanRawScore!.Value, precision: 6);
        Assert.True(result.IsAggregationAvailable);
    }

    [Fact]
    public async Task K5_TwoUnavailable_MeanOver3()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 5);
        _api.OnCompareAi = id => (id == 2 || id == 4) ? null : SuccessDto(id == 1 ? 0.3 : id == 3 ? 0.6 : 0.9);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.True(result.Success);
        Assert.Equal(5, result.ActiveReferenceCount);
        Assert.Equal(3, result.ComparedReferenceCount);
        Assert.Equal(2, result.UnavailableReferenceCount);
        var expected = (0.3 + 0.6 + 0.9) / 3.0;
        Assert.Equal(expected, result.MeanRawScore!.Value, precision: 6);
    }

    [Fact]
    public async Task ZeroAvailable_AggregationUnavailable()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 3);
        _api.OnCompareAi = _ => null;

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.False(result.IsAggregationAvailable);
        Assert.Null(result.MeanRawScore);
        Assert.Equal(3, result.ActiveReferenceCount);
        Assert.Equal(0, result.ComparedReferenceCount);
        Assert.Equal(3, result.UnavailableReferenceCount);
    }

    [Fact]
    public async Task UnavailableScore_NotTreatedAsZero()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 3);
        // 2 success with 0.5 each, 1 unavailable -> mean should be 0.5 not 0.33
        _api.OnCompareAi = id => id == 3 ? null : SuccessDto(0.5);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.True(result.IsAggregationAvailable);
        Assert.Equal(0.5, result.MeanRawScore!.Value, precision: 6);
        Assert.NotEqual(0.333333, result.MeanRawScore.Value, precision: 3);
        Assert.Equal(3, result.ActiveReferenceCount);
        Assert.Equal(2, result.ComparedReferenceCount);
        Assert.Equal(1, result.UnavailableReferenceCount);
    }

    [Fact]
    public async Task OriginalPerReferenceScoresUnchanged()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 3);
        _api.OnCompareAi = id => SuccessDto(id == 1 ? 0.11 : id == 2 ? 0.22 : 0.33);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.Equal(0.11, result.Comparisons.Single(c => c.ReferenceSignatureId == 1).Score!.Value, precision: 6);
        Assert.Equal(0.22, result.Comparisons.Single(c => c.ReferenceSignatureId == 2).Score!.Value, precision: 6);
        Assert.Equal(0.33, result.Comparisons.Single(c => c.ReferenceSignatureId == 3).Score!.Value, precision: 6);
        Assert.Equal(0.22, result.MeanRawScore!.Value, precision: 6);
    }

    [Fact]
    public async Task ComparedReferenceCount_Correct()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 4);
        _api.OnCompareAi = id => id == 4 ? null : SuccessDto(0.4);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.Equal(4, result.ActiveReferenceCount);
        Assert.Equal(3, result.ComparedReferenceCount);
    }

    [Fact]
    public async Task UnavailableReferenceCount_Correct()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 5);
        _api.OnCompareAi = id => id <= 2 ? SuccessDto(0.5) : null;

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.Equal(5, result.ActiveReferenceCount);
        Assert.Equal(2, result.ComparedReferenceCount);
        Assert.Equal(3, result.UnavailableReferenceCount);
    }

    [Fact]
    public async Task NoPersistence_Occurs()
    {
        await using var db = new CountingSaveChangesDbContext(CreateOptions());
        await SeedAsync(db, references: 3);
        _api.OnCompareAi = _ => SuccessDto(0.6);
        db.ResetSaveChangesCount();

        var result = await new VerificationService(db, _api, new FakeWebHostEnvironment { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance).CompareAiSignaturesWithReferencesAsync(1);

        Assert.True(result.Success);
        Assert.Equal(0, db.SaveChangesCount);
        Assert.False(await db.SignatureComparisons.AnyAsync());
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.False(await db.AuditLogs.AnyAsync());
        Assert.Equal(1, (await db.Cheques.SingleAsync(c => c.ChequeId == 1)).Status);
        Assert.NotNull(result.MeanRawScore);
    }

    [Fact]
    public async Task NoOpenCvCall_Introduced()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 3);
        _api.OnCompareAi = _ => SuccessDto(0.5);

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.Equal(3, _api.CompareAiCallCount);
        Assert.Equal(0, _api.CompareCallCount);
        Assert.NotNull(result.MeanRawScore);
    }

    [Fact]
    public async Task Api503_PartialFailure_PreservesBehavior_AndAggregatesOverAvailable()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 5);
        _api.OnCompareAi = id =>
        {
            if (id == 2)
            {
                return new SignatureAiComparisonResponseDto
                {
                    Success = false,
                    SimilarityScore = null,
                    Method = "ai_metric",
                    Version = "v2",
                    Model = "siamese_resnet18",
                    Device = "unavailable",
                    Message = "Modèle IA non chargé."
                };
            }
            return SuccessDto(id == 1 ? 0.2 : id == 3 ? 0.4 : id == 4 ? 0.6 : 0.8);
        };

        var result = await CreateService(db).CompareAiSignaturesWithReferencesAsync(1);

        Assert.True(result.Success);
        Assert.Equal(5, result.ActiveReferenceCount);
        Assert.Equal(4, result.ComparedReferenceCount);
        Assert.Equal(1, result.UnavailableReferenceCount);
        var expected = (0.2 + 0.4 + 0.6 + 0.8) / 4.0;
        Assert.Equal(expected, result.MeanRawScore!.Value, precision: 6);
        var unavailable = result.Comparisons.Single(c => c.ReferenceSignatureId == 2);
        Assert.False(unavailable.IsAvailable);
        Assert.Null(unavailable.Score);
        Assert.Contains("Comparaison IA indisponible", unavailable.StatusMessage);
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
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"verify-agg-db-{Guid.NewGuid():N}")
            .Options;

    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());

    private VerificationService CreateService(ChequeVerificationDbContext db)
    {
        var env = new FakeWebHostEnvironment { WebRootPath = _webRoot };
        return new VerificationService(db, _api, env, NullLogger<VerificationService>.Instance);
    }

    private async Task SeedAsync(ChequeVerificationDbContext db, int references = 2, bool createExtracted = true)
    {
        var customer = new Customer { CustomerNumber = "CUST-AI-1", FullName = "Jean Dupont", AccountNumber = "ACC-AI-1" };
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
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default) => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<ImageAnalysisResponseDto?>(null);
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureExtractionResponseDto?>(null);
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureDebugResponseDto?>(null);
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default) { CompareCallCount++; return Task.FromResult<SignatureComparisonResponseDto?>(null); }
        public async Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream es, string ef, string ec, Stream rs, string rf, string rc, CancellationToken ct = default)
        {
            CompareAiCallCount++;
            var name = Path.GetFileNameWithoutExtension(rf);
            var id = int.TryParse(name.Replace("ref", string.Empty), out var p) ? p : CompareAiCallCount;
            return await Task.FromResult(OnCompareAi?.Invoke(id));
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
        public CountingSaveChangesDbContext(DbContextOptions<ChequeVerificationDbContext> o) : base(o) { }
        public int SaveChangesCount { get; private set; }
        public void ResetSaveChangesCount() => SaveChangesCount = 0;
        public override int SaveChanges(bool a = true) { SaveChangesCount++; return base.SaveChanges(a); }
        public override Task<int> SaveChangesAsync(CancellationToken ct = default) { SaveChangesCount++; return base.SaveChangesAsync(ct); }
    }
}
