using System.Security.Cryptography;
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
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

public class SignatureReExtractionPersistenceTests : IDisposable
{
    private const string UploadsRoot = "uploads/signatures/extracted";

    private readonly string _webRoot;
    private readonly FakeVerificationApiClient _api = new();

    public SignatureReExtractionPersistenceTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "verify-retest-" + Guid.NewGuid().ToString("N"));
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
    public async Task ChequeDoesNotExist_BlocksReExtraction()
    {
        await using var db = CreateDbContext();
        _api.OnExtract = () => ValidExtractionResponse(NewPngBytes);

        var service = CreateService(db);
        var result = await service.ReExtractAndPersistSignatureAsync(chequeId: 999, userId: 1);

        Assert.False(result.Success);
        Assert.Contains("introuvable", result.Message);
        Assert.Empty(ExtractedPngFiles());
        Assert.False(await db.AuditLogs.AnyAsync());
    }

    [Fact]
    public async Task NoExistingExtractedSignature_BlocksReExtraction()
    {
        await using var db = CreateDbContext();
        await SeedChequeOnlyAsync(db);
        _api.OnExtract = () => ValidExtractionResponse(NewPngBytes);

        var service = CreateService(db);
        var result = await service.ReExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
        Assert.Contains("Exécutez d'abord", result.Message);
        Assert.False(await db.ExtractedSignatures.AnyAsync());
        Assert.Empty(ExtractedPngFiles());
        Assert.False(await db.AuditLogs.AnyAsync());
    }

    [Fact]
    public async Task VerificationResultExists_BlocksReExtraction()
    {
        await using var db = CreateDbContext();
        await SeedReExtractionAsync(db);

        db.VerificationResults.Add(new VerificationResult
        {
            ChequeId = 1,
            SimilarityScore = 0.5m,
            LowerThresholdUsed = 0.4m,
            UpperThresholdUsed = 0.8m,
            AutomaticDecision = 1,
            VerifiedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();

        _api.OnExtract = () => ValidExtractionResponse(NewPngBytes);
        var before = await db.ExtractedSignatures.AsNoTracking().SingleAsync(e => e.ChequeId == 1);

        var service = CreateService(db);
        var result = await service.ReExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
        Assert.Contains("résultat de vérification", result.Message);

        var after = await db.ExtractedSignatures.AsNoTracking().SingleAsync(e => e.ChequeId == 1);
        Assert.Equal(before.ImagePath, after.ImagePath);
        Assert.Equal(before.FileHash, after.FileHash);
        Assert.Equal(before.ExtractionConfidence, after.ExtractionConfidence);
        Assert.Single(ExtractedPngFiles());
        Assert.False(await db.AuditLogs.AnyAsync(a => a.Action == "REEXTRACT_SIGNATURE"));
    }

    [Fact]
    public async Task FastApiUnavailable_NoSqlChangeAndOldFilePreserved()
    {
        await using var db = CreateDbContext();
        await SeedReExtractionAsync(db);
        _api.OnExtract = null;

        var before = await db.ExtractedSignatures.AsNoTracking().SingleAsync(e => e.ChequeId == 1);

        var service = CreateService(db);
        var result = await service.ReExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
        Assert.Contains("n'est pas disponible", result.Message);

        var after = await db.ExtractedSignatures.AsNoTracking().SingleAsync(e => e.ChequeId == 1);
        Assert.Equal(before.ImagePath, after.ImagePath);
        Assert.Equal(before.FileHash, after.FileHash);
        Assert.Single(ExtractedPngFiles());
        Assert.False(await db.AuditLogs.AnyAsync());
    }

    [Fact]
    public async Task InvalidBase64_NoSqlChangeAndOldFilePreserved()
    {
        await using var db = CreateDbContext();
        await SeedReExtractionAsync(db);
        _api.OnExtract = () => new SignatureExtractionResponseDto
        {
            Success = true,
            OriginalWidth = 800,
            OriginalHeight = 355,
            ExtractionQuality = 0.6,
            SignatureImageBase64 = "###not-base64###"
        };

        var before = await db.ExtractedSignatures.AsNoTracking().SingleAsync(e => e.ChequeId == 1);

        var service = CreateService(db);
        var result = await service.ReExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
        Assert.Contains("Base64", result.Message);

        var after = await db.ExtractedSignatures.AsNoTracking().SingleAsync(e => e.ChequeId == 1);
        Assert.Equal(before.ImagePath, after.ImagePath);
        Assert.Equal(before.FileHash, after.FileHash);
        Assert.Single(ExtractedPngFiles());
        Assert.False(await db.AuditLogs.AnyAsync());
    }

    [Fact]
    public async Task Success_UpdatesExistingRow_OneRowOnly_NewFile_OldFileRemoved_AuditCreated()
    {
        await using var db = CreateDbContext();
        var (chequeId, originalId, oldPhysicalPath) = await SeedReExtractionAsync(db);
        _api.OnExtract = () => ValidExtractionResponse(NewPngBytes, quality: 0.6);

        var service = CreateService(db);
        var result = await service.ReExtractAndPersistSignatureAsync(chequeId, userId: 7);

        Assert.True(result.Success, result.Message);
        Assert.Equal("Signature ré-extraite avec succès.", result.Message);

        // Same ID, single row.
        var extracted = await db.ExtractedSignatures.SingleAsync(e => e.ChequeId == chequeId);
        Assert.Equal(originalId, result.ExtractedSignatureId);
        Assert.Equal(originalId, extracted.ExtractedSignatureId);
        Assert.Single(await db.ExtractedSignatures.ToListAsync());

        // New path (different from the old one), server-side GUID, new quality,
        // new hash and updated timestamp.
        Assert.StartsWith($"/{UploadsRoot}/", result.ImagePath);
        Assert.Equal(result.ImagePath, extracted.ImagePath);
        Assert.NotEqual($"/{UploadsRoot}/old.png", extracted.ImagePath);
        Assert.Equal((decimal)0.6, extracted.ExtractionConfidence);
        Assert.Equal(0.6m, result.ExtractionQuality);

        var file = Assert.Single(ExtractedPngFiles());
        Assert.True(Guid.TryParse(Path.GetFileName(file).Replace(".png", ""), out _), "Nom de fichier doit être un GUID.");
        var savedBytes = await File.ReadAllBytesAsync(file);
        var expectedHash = Convert.ToHexString(SHA256.HashData(savedBytes)).ToLowerInvariant();
        Assert.Equal(expectedHash, extracted.FileHash);

        var oldExtractedAt = DateTime.UtcNow.AddDays(-1);
        Assert.True(extracted.ExtractedAt > oldExtractedAt);

        // Old physical crop deleted, new crop exists.
        Assert.False(File.Exists(oldPhysicalPath));

        // Cheque status preserved (never Verified, no new value).
        var cheque = await db.Cheques.SingleAsync(c => c.ChequeId == chequeId);
        Assert.Equal(2, cheque.Status);
        Assert.False(await db.VerificationResults.AnyAsync(v => v.ChequeId == chequeId));

        // AuditLog REEXTRACT_SIGNATURE.
        var audit = await db.AuditLogs.SingleAsync(a => a.Action == "REEXTRACT_SIGNATURE");
        Assert.Equal(nameof(ExtractedSignature), audit.EntityName);
        Assert.Equal(extracted.ExtractedSignatureId, audit.EntityId);
        Assert.Equal(7, audit.UserId);
        Assert.Contains("CHQ-0001", audit.Description);
    }

    [Fact]
    public async Task SqlErrorAfterNewFileWrite_Rollback_NewFileDeleted_OldRowAndOldFileUntouched()
    {
        await using var db = new FailingSaveChangesDbContext(CreateOptions());
        await SeedReExtractionAsync(db);
        db.FailNextSave = true;
        _api.OnExtract = () => ValidExtractionResponse(NewPngBytes);

        var before = await db.ExtractedSignatures.AsNoTracking().SingleAsync(e => e.ChequeId == 1);

        var service = CreateService(db);
        var result = await service.ReExtractAndPersistSignatureAsync(1, 1);

        Assert.False(result.Success);
        Assert.Contains("ré-extraction", result.Message);

        // Old DB row untouched (AsNoTracking forces the stored values).
        var after = await db.ExtractedSignatures.AsNoTracking().SingleAsync(e => e.ChequeId == 1);
        Assert.Equal(before.ImagePath, after.ImagePath);
        Assert.Equal(before.FileHash, after.FileHash);
        Assert.Equal(before.ExtractionConfidence, after.ExtractionConfidence);

        // Only the old file remains; the new file was cleaned up.
        var files = ExtractedPngFiles();
        Assert.Single(files);
        Assert.EndsWith("old.png", files[0]);
        Assert.False(await db.AuditLogs.AnyAsync());
    }

    [Fact]
    public async Task OldFileDeletionFails_AfterCommit_UpdateStaysCommitted_AndWarningLogged()
    {
        await using var db = CreateDbContext();
        var (chequeId, _, oldPhysicalPath) = await SeedReExtractionAsync(db);
        _api.OnExtract = () => ValidExtractionResponse(NewPngBytes);
        var logger = new ListLogger<VerificationService>();

        var service = CreateService(db, logger);
        var result = await new DeferredServiceRunner(service).RunAsync(chequeId, oldPhysicalPath);

        Assert.True(result.Success, result.Message);

        // DB update committed with the new path.
        var extracted = await db.ExtractedSignatures.AsNoTracking().SingleAsync(e => e.ChequeId == chequeId);
        Assert.NotEqual($"/{UploadsRoot}/old.png", extracted.ImagePath);
        Assert.StartsWith($"/{UploadsRoot}/", extracted.ImagePath);

        // New file exists; old file still present (deletion was blocked).
        var newPhysical = Path.Combine(_webRoot, extracted.ImagePath.TrimStart('/').Replace('/', Path.DirectorySeparatorChar));
        Assert.True(File.Exists(newPhysical));
        Assert.True(File.Exists(oldPhysicalPath));

        // A warning was logged about the failed deletion.
        Assert.Contains(logger.Entries, e => e.Level == LogLevel.Warning && e.Message.Contains("ancien", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task MultipleReExtractions_NeverCreateASecondRow()
    {
        await using var db = CreateDbContext();
        var (chequeId, originalId, _) = await SeedReExtractionAsync(db);

        _api.Responses = new Queue<SignatureExtractionResponseDto?>();
        _api.Responses.Enqueue(ValidExtractionResponse(NewPngBytes, quality: 0.6));
        _api.Responses.Enqueue(ValidExtractionResponse(NewerPngBytes, quality: 0.7));

        var service = CreateService(db);

        var first = await service.ReExtractAndPersistSignatureAsync(chequeId, 1);
        var second = await service.ReExtractAndPersistSignatureAsync(chequeId, 1);

        Assert.True(first.Success);
        Assert.True(second.Success);

        var rows = await db.ExtractedSignatures.ToListAsync();
        var single = Assert.Single(rows);
        Assert.Equal(originalId, single.ExtractedSignatureId);
        Assert.Equal(originalId, first.ExtractedSignatureId);
        Assert.Equal(originalId, second.ExtractedSignatureId);
        Assert.Equal(second.ImagePath, single.ImagePath);
        Assert.Equal((decimal)0.7, single.ExtractionConfidence);
        Assert.Equal(2, await db.AuditLogs.CountAsync(a => a.Action == "REEXTRACT_SIGNATURE"));
        Assert.Single(ExtractedPngFiles());
    }

    [Fact]
    public void Controller_ReExtractAction_HasPostAntiforgeryAndAuthorization()
    {
        var type = typeof(VerificationsController);

        // Class level requires authentication; role restriction is explicit
        // per action (class-level Roles would stack with and override wider
        // read-only policies such as History).
        var controllerAuthorize = type.GetCustomAttributes(typeof(AuthorizeAttribute), inherit: true)
            .Cast<AuthorizeAttribute>()
            .FirstOrDefault();
        Assert.NotNull(controllerAuthorize);

        var action = type.GetMethod("ReExtractSignature");
        Assert.NotNull(action);
        Assert.NotNull(action!.GetCustomAttributes(typeof(HttpPostAttribute), inherit: false).FirstOrDefault());
        Assert.NotNull(action.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), inherit: false).FirstOrDefault());

        var actionAuthorize = action.GetCustomAttributes(typeof(AuthorizeAttribute), inherit: false)
            .Cast<AuthorizeAttribute>()
            .FirstOrDefault();
        Assert.NotNull(actionAuthorize);
        Assert.Contains("Utilisateur", actionAuthorize!.Roles);
        Assert.Contains("Administrateur", actionAuthorize.Roles);
    }

    private sealed class DeferredServiceRunner
    {
        private readonly VerificationService _service;
        public DeferredServiceRunner(VerificationService service) => _service = service;

        public async Task<SignatureExtractionOperationResult> RunAsync(int chequeId, string oldPhysicalPath)
        {
            // Lock the old file (FileShare.None) so its deletion throws
            // IOException after the commit.
            using var lockHandle = new FileStream(oldPhysicalPath, FileMode.Open, FileAccess.Read, FileShare.None);
            return await _service.ReExtractAndPersistSignatureAsync(chequeId, userId: 1);
        }
    }

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
    {
        return new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"verify-redb-{Guid.NewGuid():N}")
            .Options;
    }

    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());

    private VerificationService CreateService(ChequeVerificationDbContext db, ILogger<VerificationService>? logger = null)
    {
        var env = new FakeWebHostEnvironment { WebRootPath = _webRoot };
        return new VerificationService(db, _api, env, logger ?? NullLogger<VerificationService>.Instance);
    }

    private async Task SeedChequeOnlyAsync(ChequeVerificationDbContext db)
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
            ImagePath = "/placeholder.png",
            Status = 2,
            UploadedAt = DateTime.UtcNow
        });

        await db.SaveChangesAsync();

        var cheque = await db.Cheques.SingleAsync(c => c.ChequeNumber == "CHQ-0001");
        var chequePath = Path.Combine(Path.GetTempPath(), "verify-tests-cheque-" + Guid.NewGuid().ToString("N") + ".png");
        await File.WriteAllBytesAsync(chequePath, OldPngBytes);
        cheque.ImagePath = chequePath;
        await db.SaveChangesAsync();
    }

    private async Task<(int ChequeId, int ExtractedSignatureId, string OldPhysicalPath)> SeedReExtractionAsync(ChequeVerificationDbContext db)
    {
        await SeedChequeOnlyAsync(db);
        var cheque = await db.Cheques.SingleAsync(c => c.ChequeNumber == "CHQ-0001");

        db.ExtractedSignatures.Add(new ExtractedSignature
        {
            ChequeId = cheque.ChequeId,
            ImagePath = $"/{UploadsRoot}/old.png",
            FileHash = ComputeSha256Hex(OldPngBytes),
            ExtractionConfidence = 0.8948m,
            ExtractedAt = DateTime.UtcNow.AddDays(-1)
        });
        await db.SaveChangesAsync();

        var extracted = await db.ExtractedSignatures.SingleAsync(e => e.ChequeId == cheque.ChequeId);
        var oldPhysical = Path.Combine(_webRoot, UploadsRoot.Replace('/', Path.DirectorySeparatorChar), "old.png");
        Directory.CreateDirectory(Path.GetDirectoryName(oldPhysical)!);
        await File.WriteAllBytesAsync(oldPhysical, OldPngBytes);

        return (cheque.ChequeId, extracted.ExtractedSignatureId, oldPhysical);
    }

    private static SignatureExtractionResponseDto ValidExtractionResponse(byte[] pngBytes, double quality = 0.6) => new()
    {
        Success = true,
        OriginalWidth = 800,
        OriginalHeight = 355,
        CandidateRoi = new BoundingBoxDto { X = 400, Y = 195, Width = 384, Height = 142 },
        SignatureBbox = new BoundingBoxDto { X = 128, Y = 30, Width = 185, Height = 91 },
        ExtractionQuality = quality,
        ImageFormat = "png",
        SignatureImageBase64 = Convert.ToBase64String(pngBytes)
    };

    private static string ComputeSha256Hex(byte[] bytes)
        => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private const string TinyPngBase64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    private static readonly byte[] OldPngBytes = Convert.FromBase64String(TinyPngBase64);
    private static readonly byte[] NewPngBytes = ToggleLastByte(OldPngBytes, 0x01);
    private static readonly byte[] NewerPngBytes = ToggleLastByte(OldPngBytes, 0x02);

    private static byte[] ToggleLastByte(byte[] source, byte xor)
    {
        var copy = new byte[source.Length];
        source.CopyTo(copy, 0);
        copy[^1] ^= xor;
        return copy;
    }

    private sealed class FakeVerificationApiClient : IVerificationApiClient
    {
        public Queue<SignatureExtractionResponseDto?>? Responses { get; set; }
        public Func<SignatureExtractionResponseDto?>? OnExtract { get; set; }

        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken cancellationToken = default)
            => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });

        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<ImageAnalysisResponseDto?>(null);

        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult(Responses is { Count: > 0 } ? Responses.Dequeue() : OnExtract?.Invoke());

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

    private sealed class ListLogger<T> : ILogger<T>
    {
        public List<(LogLevel Level, string Message)> Entries { get; } = new();

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            Entries.Add((logLevel, formatter(state, exception)));
        }
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
