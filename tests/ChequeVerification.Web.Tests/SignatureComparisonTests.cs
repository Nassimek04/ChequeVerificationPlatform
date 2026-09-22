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
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

public class SignatureComparisonTests : IDisposable
{
    private readonly string _webRoot;
    private readonly FakeVerificationApiClient _api = new();

    public SignatureComparisonTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "verify-compare-" + Guid.NewGuid().ToString("N"));
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
        _api.OnCompare = _ => new SignatureComparisonResponseDto { Success = true, SimilarityScore = 0.9 };

        var service = CreateService(db);
        var result = await service.CompareSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.Contains("Aucune signature n'a été extraite", result.Message);
        Assert.Empty(result.Comparisons);
        Assert.Equal(0, _api.CompareCallCount);
    }

    [Fact]
    public async Task NoActiveReference_BlocksOperation()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 0);
        _api.OnCompare = _ => new SignatureComparisonResponseDto { Success = true, SimilarityScore = 0.9 };

        var service = CreateService(db);
        var result = await service.CompareSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.Contains("Aucune signature de référence active", result.Message);
        Assert.Empty(result.Comparisons);
    }

    [Fact]
    public async Task ExtractedFileMissing_BlocksOperation()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        File.Delete(_webRoot + "\\extracted.png");
        _api.OnCompare = _ => new SignatureComparisonResponseDto { Success = true, SimilarityScore = 0.9 };

        var service = CreateService(db);
        var result = await service.CompareSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.Contains("introuvable", result.Message);
        Assert.Empty(result.Comparisons);
    }

    [Fact]
    public async Task MissingReferencePhysicalFile_IsMarkedUnavailable_OthersCompared()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 2);
        // Reference #2 points to a physical file that does not exist.
        var ref2 = await db.ReferenceSignatures.SingleAsync(r => r.ReferenceSignatureId == 2);
        ref2.ImagePath = _webRoot + "\\missing-ref.png";
        await db.SaveChangesAsync();

        _api.OnCompare = _ => new SignatureComparisonResponseDto
        {
            Success = true,
            SimilarityScore = 0.75,
            Method = "opencv_baseline",
            Version = "v1"
        };

        var service = CreateService(db);
        var result = await service.CompareSignaturesWithReferencesAsync(1);

        Assert.True(result.Success, result.Message);
        Assert.Equal(2, result.Comparisons.Count);

        var compared = result.Comparisons.Single(c => c.ReferenceSignatureId == 1);
        Assert.True(compared.IsAvailable);
        Assert.Equal(0.75, compared.Score);
        Assert.Equal("opencv_baseline", compared.Method);
        Assert.Equal("v1", compared.Version);

        var unavailable = result.Comparisons.Single(c => c.ReferenceSignatureId == 2);
        Assert.False(unavailable.IsAvailable);
        Assert.Null(unavailable.Score);
        Assert.Contains("introuvable", unavailable.StatusMessage);

        Assert.Equal(1, result.BestReferenceSignatureId);
        Assert.Equal(0.75, result.BestSimilarityScore);
    }

    [Fact]
    public async Task MultipleReferences_AllCompared_BestSelected()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 3);
        _api.OnCompare = referenceId => new SignatureComparisonResponseDto
        {
            Success = true,
            SimilarityScore = referenceId == 2 ? 0.92 : 0.5,
            Method = "opencv_baseline",
            Version = "v1"
        };

        var service = CreateService(db);
        var result = await service.CompareSignaturesWithReferencesAsync(1);

        Assert.True(result.Success);
        Assert.Equal(3, result.Comparisons.Count);
        Assert.All(result.Comparisons, c => Assert.True(c.IsAvailable));
        Assert.Equal(2, result.BestReferenceSignatureId);
        Assert.Equal(0.92, result.BestSimilarityScore);
        Assert.Equal(3, _api.CompareCallCount);
    }

    [Fact]
    public async Task ApiDown_NoCrash_ComparisonsMarkedUnavailable()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 2);
        _api.OnCompare = _ => null;

        var service = CreateService(db);
        var result = await service.CompareSignaturesWithReferencesAsync(1);

        Assert.False(result.Success);
        Assert.Contains("indisponible", result.Message);
        Assert.Equal(2, result.Comparisons.Count);
        Assert.All(result.Comparisons, c =>
        {
            Assert.False(c.IsAvailable);
            Assert.Null(c.Score);
        });
        Assert.Null(result.BestReferenceSignatureId);
    }

    [Fact]
    public async Task Comparison_IsReadOnly_NoSqlChangeNoAuditNoStatusChange()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, references: 2);
        _api.OnCompare = _ => new SignatureComparisonResponseDto
        {
            Success = true,
            SimilarityScore = 0.8,
            Method = "opencv_baseline",
            Version = "v1"
        };

        var service = CreateService(db);
        var result = await service.CompareSignaturesWithReferencesAsync(1);

        Assert.True(result.Success);

        // No new rows, no status change, no audit entry.
        Assert.Single(await db.ExtractedSignatures.ToListAsync());
        Assert.Equal(2, await db.ReferenceSignatures.CountAsync());
        Assert.False(await db.AuditLogs.AnyAsync());
        Assert.False(await db.SignatureComparisons.AnyAsync());
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.Equal(1, (await db.Cheques.SingleAsync(c => c.ChequeId == 1)).Status);
    }

    [Fact]
    public async Task TestSignatureComparisonAction_IsReadOnly_ZeroSaveChanges_ZeroRows_BestInMemory()
    {
        await using var db = new CountingSaveChangesDbContext(CreateOptions());
        await SeedAsync(db, references: 3);
        _api.OnCompare = referenceId => new SignatureComparisonResponseDto
        {
            Success = true,
            SimilarityScore = referenceId == 2 ? 0.92 : 0.5,
            Method = "opencv_baseline",
            Version = "v1"
        };

        // Seeding has already flushed; from now on every SaveChanges counts.
        db.ResetSaveChangesCount();

        var service = CreateService(db);
        var controller = new VerificationsController(
            service,
            _api,
            new FakeWebHostEnvironment { WebRootPath = _webRoot });

        var actionResult = await controller.TestSignatureComparison(1);
        var viewResult = Assert.IsType<ViewResult>(actionResult);
        var model = Assert.IsType<VerificationCreateViewModel>(viewResult.Model);

        Assert.True(model.SignatureComparison!.Success);
        Assert.Equal(3, model.SignatureComparison.Comparisons.Count);
        Assert.Equal(2, model.SignatureComparison.BestReferenceSignatureId);
        Assert.Equal(0.92, model.SignatureComparison.BestSimilarityScore);

        // The experimental test workflow performs ZERO SQL writes.
        Assert.Equal(0, db.SaveChangesCount);
        Assert.False(await db.SignatureComparisons.AnyAsync());
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.False(await db.AuditLogs.AnyAsync());
        Assert.Single(await db.ExtractedSignatures.ToListAsync());
        Assert.Equal(1, (await db.Cheques.SingleAsync(c => c.ChequeId == 1)).Status);
    }

    [Fact]
    public void Dto_SnakeCaseProperties_Deserialized()
    {
        var json = """
            {
              "success": true,
              "similarity_score": 0.871234,
              "method": "opencv_baseline",
              "version": "v1",
              "metrics": {
                "mask_overlap": 0.9,
                "normalized_correlation": 0.85,
                "density_similarity": 0.95
              }
            }
            """;

        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web);
        var dto = JsonSerializer.Deserialize<SignatureComparisonResponseDto>(json, options);

        Assert.NotNull(dto);
        Assert.True(dto!.Success);
        Assert.Equal(0.871234, dto.SimilarityScore);
        Assert.Equal("opencv_baseline", dto.Method);
        Assert.Equal("v1", dto.Version);
        Assert.NotNull(dto.Metrics);
        Assert.Equal(0.9, dto.Metrics!.MaskOverlap);
        Assert.Equal(0.85, dto.Metrics.NormalizedCorrelation);
        Assert.Equal(0.95, dto.Metrics.DensitySimilarity);
    }

    [Fact]
    public void Controller_HasAuthorizationAndActionHasAntiforgery()
    {
        var type = typeof(VerificationsController);
        // Class level requires authentication; role restriction is explicit
        // per action (class-level Roles would stack with and override wider
        // read-only policies such as History).
        var controllerAuthorize = type.GetCustomAttributes(typeof(AuthorizeAttribute), inherit: true)
            .Cast<AuthorizeAttribute>()
            .FirstOrDefault();

        Assert.NotNull(controllerAuthorize);

        var action = type.GetMethod("TestSignatureComparison");
        Assert.NotNull(action);
        Assert.NotNull(action!.GetCustomAttributes(typeof(HttpPostAttribute), inherit: false).FirstOrDefault());
        Assert.NotNull(action.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), inherit: false).FirstOrDefault());
        var actionAuthorize = action.GetCustomAttributes(typeof(AuthorizeAttribute), inherit: false)
            .Cast<AuthorizeAttribute>()
            .FirstOrDefault();
        Assert.NotNull(actionAuthorize);
        Assert.Contains("Utilisateur", actionAuthorize!.Roles);
        Assert.Contains("Administrateur", actionAuthorize.Roles);
        Assert.DoesNotContain("Contrôleur", actionAuthorize.Roles!.Split(','));
    }

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
    {
        return new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"verify-compare-db-{Guid.NewGuid():N}")
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
            CustomerNumber = "CUST-1",
            FullName = "Jean Dupont",
            AccountNumber = "ACC-1"
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
            ChequeNumber = "CHQ-0001",
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

        // Physical cheque image referenced above.
        await File.WriteAllBytesAsync(_webRoot + "\\cheque1.png", new byte[] { 1, 1, 1 });
    }

    private sealed class FakeVerificationApiClient : IVerificationApiClient
    {
        public Func<int, SignatureComparisonResponseDto?>? OnCompare { get; set; }
        public int CompareCallCount { get; private set; }

        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken cancellationToken = default)
            => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });

        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<ImageAnalysisResponseDto?>(null);

        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<SignatureExtractionResponseDto?>(null);

        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<SignatureDebugResponseDto?>(null);

        public async Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default)
        {
            CompareCallCount++;
            return await Task.FromResult(OnCompare?.Invoke(CompareCallCount));
        }

        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream a, string b, string c, CancellationToken ct = default) => Task.FromResult<ChequeOcrResponseDto?>(null);
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default)
            => Task.FromResult<SignatureAiComparisonResponseDto?>(null);
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