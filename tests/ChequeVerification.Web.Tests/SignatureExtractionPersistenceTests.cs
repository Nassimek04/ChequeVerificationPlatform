using System.Security.Cryptography;
using System.Text;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Controllers;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

public class SignatureExtractionPersistenceTests : IDisposable
{
    private const string UploadsRoot = "uploads/signatures/extracted";

    private readonly string _webRoot;
    private readonly FakeVerificationApiClient _api = new();

    public SignatureExtractionPersistenceTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "verify-tests-" + Guid.NewGuid().ToString("N"));
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

    private string[] ExtractedPngFiles()
    {
        var directory = Path.Combine(_webRoot, UploadsRoot);
        return Directory.Exists(directory) ? Directory.GetFiles(directory, "*.png") : Array.Empty<string>();
    }

    [Fact]
    public async Task FastApiUnavailable_NoSqlChangeAndNoFile()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        _api.OnExtract = null;

        var service = CreateService(db);
        var result = await service.ExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
        Assert.False(await db.ExtractedSignatures.AnyAsync());
        Assert.False(await db.AuditLogs.AnyAsync());
        Assert.Empty(ExtractedPngFiles());
    }

    [Fact]
    public async Task AlreadyExtracted_BlocksOperation()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        db.ExtractedSignatures.Add(new ExtractedSignature
        {
            ChequeId = 1,
            ImagePath = $"/{UploadsRoot}/existing.png",
            FileHash = "abc",
            ExtractedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();
        _api.OnExtract = () => ValidExtractionResponse();

        var service = CreateService(db);
        var result = await service.ExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
Assert.Contains("déjà été extraite", result.Message);
        Assert.Empty(ExtractedPngFiles());
    }

    [Fact]
    public async Task InvalidBase64_NoSqlChangeAndNoFile()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        _api.OnExtract = () => new SignatureExtractionResponseDto
        {
            Success = true,
            OriginalWidth = 800,
            OriginalHeight = 355,
            ExtractionQuality = 0.6,
            SignatureImageBase64 = "###not-base64###"
        };

        var service = CreateService(db);
        var result = await service.ExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
        Assert.False(await db.ExtractedSignatures.AnyAsync());
        Assert.False(await db.AuditLogs.AnyAsync());
        Assert.Empty(ExtractedPngFiles());
    }

    [Fact]
    public async Task EmptyBase64_NoSqlChangeAndNoFile()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        _api.OnExtract = () => new SignatureExtractionResponseDto
        {
            Success = true,
            ExtractionQuality = 0.5,
            SignatureImageBase64 = ""
        };

        var service = CreateService(db);
        var result = await service.ExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
        Assert.False(await db.ExtractedSignatures.AnyAsync());
    }

    [Fact]
    public async Task Success_CreatesFileAndEntity_StatusTwo_FileHashAndAudit()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        _api.OnExtract = () => ValidExtractionResponse();

        var service = CreateService(db);
        var result = await service.ExtractAndPersistSignatureAsync(1, userId: 7);

        Assert.True(result.Success, result.Message);
        Assert.Equal("Signature extraite avec succès.", result.Message);
        Assert.True(result.ExtractedSignatureId > 0);
        Assert.StartsWith($"/{UploadsRoot}/", result.ImagePath);
        Assert.InRange(result.ExtractionQuality ?? 0, 0m, 1m);

        // File persisted with a server-generated GUID name.
        var files = ExtractedPngFiles();
        var file = Assert.Single(files);
        var fileName = Path.GetFileName(file);
        Assert.True(Guid.TryParse(fileName.Replace(".png", ""), out _), "Nom de fichier doit être un GUID.");

        // Entity created.
        var extracted = await db.ExtractedSignatures.SingleAsync(e => e.ChequeId == 1);
        Assert.Equal(result.ExtractedSignatureId, extracted.ExtractedSignatureId);
        Assert.Equal(result.ImagePath, extracted.ImagePath);
        Assert.Equal((decimal)0.6, extracted.ExtractionConfidence);
        Assert.NotNull(extracted.FileHash);

        // SHA-256 over the actually saved bytes, hex representation.
        var savedBytes = await File.ReadAllBytesAsync(file);
        var expectedHash = Convert.ToHexString(SHA256.HashData(savedBytes)).ToLowerInvariant();
        Assert.Equal(expectedHash, extracted.FileHash);

        // Cheque status -> 2 (En traitement).
        var cheque = await db.Cheques.SingleAsync(c => c.ChequeId == 1);
        Assert.Equal(2, cheque.Status);

        // AuditLog created.
        var audit = await db.AuditLogs.SingleAsync();
        Assert.Equal("EXTRACT_SIGNATURE", audit.Action);
        Assert.Equal(nameof(ExtractedSignature), audit.EntityName);
        Assert.Equal(extracted.ExtractedSignatureId, audit.EntityId);
        Assert.Equal(7, audit.UserId);
        Assert.Contains("CHQ-0001", audit.Description);
    }

    [Fact]
    public async Task DoubleClick_SecondCallBlocked_OnlyOneFile()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db);
        _api.OnExtract = () => ValidExtractionResponse();

        var service = CreateService(db);
        var first = await service.ExtractAndPersistSignatureAsync(1, 1);
        var second = await service.ExtractAndPersistSignatureAsync(1, 1);

        Assert.True(first.Success);
        Assert.False(second.Success);
Assert.Contains("déjà été extraite", second.Message);
        Assert.Single(await db.ExtractedSignatures.ToListAsync());
        Assert.Single(ExtractedPngFiles());
    }

    [Fact]
    public async Task SqlErrorAfterFileWrite_FileIsRemoved()
    {
        await using var db = new FailingSaveChangesDbContext(CreateOptions());
        await SeedAsync(db);
        db.FailNextSave = true;
        _api.OnExtract = () => ValidExtractionResponse();

        var service = CreateService(db);
        var result = await service.ExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
        // No orphan file left behind.
        Assert.Empty(ExtractedPngFiles());
        Assert.False(await db.ExtractedSignatures.AnyAsync());
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

        var action = type.GetMethod("ExtractSignature");
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
            .UseInMemoryDatabase($"verify-db-{Guid.NewGuid():N}")
            .Options;
    }

    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());

    private VerificationService CreateService(ChequeVerificationDbContext db)
    {
        var env = new FakeWebHostEnvironment { WebRootPath = _webRoot };
        return new VerificationService(db, _api, env, NullLogger<VerificationService>.Instance);
    }

    private static async Task SeedAsync(ChequeVerificationDbContext db)
    {
        var customer = new Customer
        {
            CustomerNumber = "CUST-1",
            FullName = "Jean Dupont",
            AccountNumber = "ACC-1"
        };
        db.Customers.Add(customer);

        db.ReferenceSignatures.Add(new ReferenceSignature
        {
            CustomerId = customer.CustomerId,
            ImagePath = "/uploads/reference/ref1.png",
            CreatedAt = DateTime.UtcNow,
            IsActive = true
        });

        db.Cheques.Add(new Cheque
        {
            CustomerId = customer.CustomerId,
            ImportedByUserId = 1,
            ChequeNumber = "CHQ-0001",
            ImagePath = "/uploads/cheques/cheque1.png",
            Status = 1,
            UploadedAt = DateTime.UtcNow
        });

        await db.SaveChangesAsync();

        // Create the physical cheque image referenced above.
        var chequePath = Path.Combine(Path.GetTempPath(), "verify-tests-cheque-" + Guid.NewGuid().ToString("N") + ".png");
        await File.WriteAllBytesAsync(chequePath, Convert.FromBase64String(TinyPngBase64));
        var cheque = await db.Cheques.SingleAsync(c => c.ChequeNumber == "CHQ-0001");
        cheque.ImagePath = chequePath;
        await db.SaveChangesAsync();
    }

    private static SignatureExtractionResponseDto ValidExtractionResponse() => new()
    {
        Success = true,
        OriginalWidth = 800,
        OriginalHeight = 355,
        CandidateRoi = new BoundingBoxDto { X = 400, Y = 195, Width = 384, Height = 142 },
        SignatureBbox = new BoundingBoxDto { X = 128, Y = 30, Width = 185, Height = 91 },
        ExtractionQuality = 0.6,
        ImageFormat = "png",
        SignatureImageBase64 = TinyPngBase64
    };

    private const string TinyPngBase64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    private sealed class FakeVerificationApiClient : IVerificationApiClient
    {
        public Func<SignatureExtractionResponseDto?>? OnExtract { get; set; }

        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken cancellationToken = default)
            => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });

        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<ImageAnalysisResponseDto?>(null);

        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult(OnExtract?.Invoke());

public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<SignatureDebugResponseDto?>(null);

        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default)
            => Task.FromResult<SignatureComparisonResponseDto?>(null);

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

    private sealed class FailingSaveChangesDbContext : ChequeVerificationDbContext
    {
        public FailingSaveChangesDbContext(DbContextOptions<ChequeVerificationDbContext> options)
            : base(options)
        {
        }

        public bool FailNextSave { get; set; }

        public override Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
        {
            if (FailNextSave)
            {
                FailNextSave = false;
                throw new DbUpdateException("Erreur SQL simulée.");
            }

            return base.SaveChangesAsync(cancellationToken);
        }
    }
}
