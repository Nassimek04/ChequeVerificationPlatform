using System.Security.Cryptography;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Cheques;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

/// <summary>
/// Source-file safety: user-selected cheque images are INPUTS. Preview,
/// confirm, failure, cancellation and cleanup must never mutate, move or
/// delete the original source files; staging/cleanup stays inside the
/// application-owned webRoot.
/// </summary>
public class BatchImportSourceSafetyTests : IDisposable
{
    private readonly string _webRoot;
    private readonly string _sourceDir;

    public BatchImportSourceSafetyTests()
    {
        var tag = Guid.NewGuid().ToString("N");
        _webRoot = Path.Combine(Path.GetTempPath(), "srcsafe-webroot-" + tag);
        _sourceDir = Path.Combine(Path.GetTempPath(), "srcsafe-source-" + tag);
        Directory.CreateDirectory(_webRoot);
        Directory.CreateDirectory(_sourceDir);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { }
        try { if (Directory.Exists(_sourceDir)) Directory.Delete(_sourceDir, true); } catch { }
    }

    // ---------- helpers ----------

    private static byte[] MinimalJpeg()
    {
        var bytes = new List<byte> { 0xFF, 0xD8, 0xFF, 0xE0 };
        bytes.AddRange(new byte[200]);
        return bytes.ToArray();
    }

    /// <summary>Simulates a user-selected source file, then a browser upload
    /// (bytes read into memory; the source file stays on disk).</summary>
    private string WriteSourceFile(string name, byte[] content)
    {
        var path = Path.Combine(_sourceDir, name);
        File.WriteAllBytes(path, content);
        return path;
    }

    private static IFormFile UploadOf(string sourcePath, string contentType)
    {
        var bytes = File.ReadAllBytes(sourcePath); // read/copy, never move
        return new FormFile(new MemoryStream(bytes), 0, bytes.Length, "Files",
            Path.GetFileName(sourcePath))
        {
            Headers = new HeaderDictionary(),
            ContentType = contentType
        };
    }

    private static string Sha256Of(string path)
    {
        using var sha = SHA256.Create();
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(sha.ComputeHash(stream));
    }

    private DbContextOptions<ChequeVerificationDbContext> Options()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"srcsafe-db-{Guid.NewGuid():N}").Options;

    private static ChequeOcrResponseDto OcrOk(string cheque, string? account, string? amount)
        => new()
        {
            Success = true,
            Message = "OCR effectuée avec succès.",
            FullText = $"{cheque} {account}",
            Lines = new(),
            Fields = new ChequeOcrFieldsDto
            {
                ChequeNumber = cheque,
                AccountNumber = account,
                AmountNumeric = amount
            },
            ProcessingMs = 5,
            Lang = "fr",
            Device = "cpu"
        };

    private sealed class FakeOcr : IVerificationApiClient
    {
        public ChequeOcrResponseDto? Default { get; set; }
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default) => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<ImageAnalysisResponseDto?>(null);
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureExtractionResponseDto?>(null);
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureDebugResponseDto?>(null);
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default) => Task.FromResult<SignatureComparisonResponseDto?>(null);
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default) => Task.FromResult<SignatureAiComparisonResponseDto?>(null);
        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult(Default);
    }

    private async Task SeedCustomerAsync(ChequeVerificationDbContext db, string account = "BLIND-0001")
    {
        db.Customers.Add(new Customer
        {
            CustomerNumber = "CUST-SRC",
            FullName = "Source Safety",
            AccountNumber = account,
            CreatedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();
    }

    private ChequeBatchImportService Service(ChequeVerificationDbContext db, IVerificationApiClient api)
        => new(db, api, NullLogger<ChequeBatchImportService>.Instance);

    private static BatchImportConfirmViewModel ToConfirm(BatchImportPreviewViewModel preview)
        => new()
        {
            Rows = preview.Rows.Select(r => new BatchImportConfirmRowInput
            {
                TempToken = r.TempToken,
                OriginalFileName = r.OriginalFileName,
                MappingSource = r.MappingSource,
                ChequeNumber = r.ChequeNumber,
                AccountNumber = r.AccountNumber,
                CorrectedAmount = r.AmountEdit,
                OcrAmount = r.Amount.HasValue
                    ? r.Amount.Value.ToString(System.Globalization.CultureInfo.InvariantCulture)
                    : null
            }).ToList()
        };

    // ---------- 1: preview does not mutate source ----------

    [Fact]
    public async Task Preview_DoesNotMutateSourceFile()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomerAsync(db);
        var source = WriteSourceFile("cheque-src.jpg", MinimalJpeg());
        var before = Sha256Of(source);
        var api = new FakeOcr { Default = OcrOk("CHQ-SRC-001", "BLIND-0001", "100.00") };

        await Service(db, api).BuildPreviewAsync(
            new List<IFormFile> { UploadOf(source, "image/jpeg") }, _webRoot);

        Assert.True(File.Exists(source));
        Assert.Equal(before, Sha256Of(source));
    }

    // ---------- 2: confirm does not mutate source ----------

    [Fact]
    public async Task Confirm_DoesNotMutateSourceFile()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomerAsync(db);
        var source = WriteSourceFile("cheque-src.jpg", MinimalJpeg());
        var before = Sha256Of(source);
        var api = new FakeOcr { Default = OcrOk("CHQ-SRC-001", "BLIND-0001", "100.00") };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { UploadOf(source, "image/jpeg") }, _webRoot);
        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 1, _webRoot);

        Assert.Equal(1, result.ImportedCount);
        Assert.True(File.Exists(source));
        Assert.Equal(before, Sha256Of(source));
    }

    // ---------- 3: failed row does not mutate source ----------

    [Fact]
    public async Task FailedRow_DoesNotMutateSourceFile()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomerAsync(db);
        var source = WriteSourceFile("cheque-src.jpg", MinimalJpeg());
        var before = Sha256Of(source);
        var api = new FakeOcr { Default = OcrOk("CHQ-SRC-001", "UNKNOWN-999", "100.00") };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { UploadOf(source, "image/jpeg") }, _webRoot);
        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 1, _webRoot);

        Assert.Equal(0, result.ImportedCount);
        Assert.True(File.Exists(source));
        Assert.Equal(before, Sha256Of(source));
    }

    // ---------- 4: cancel (no confirm) does not mutate source ----------

    [Fact]
    public async Task Cancel_DoesNotMutateSourceFile()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomerAsync(db);
        var source = WriteSourceFile("cheque-src.jpg", MinimalJpeg());
        var before = Sha256Of(source);
        var api = new FakeOcr { Default = OcrOk("CHQ-SRC-001", "BLIND-0001", "100.00") };

        var preview = await Service(db, api).BuildPreviewAsync(
            new List<IFormFile> { UploadOf(source, "image/jpeg") }, _webRoot);

        // User cancels: no ConfirmImportAsync call. Staging happened in webRoot.
        Assert.Single(preview.Rows);
        Assert.True(File.Exists(source));
        Assert.Equal(before, Sha256Of(source));
        Assert.DoesNotContain(Directory.GetFiles(_sourceDir),
            f => f.Contains("batch-temp", StringComparison.OrdinalIgnoreCase));
    }

    // ---------- 5: mixed batch partial success keeps every source ----------

    [Fact]
    public async Task MixedBatch_PreservesAllSourceFiles()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomerAsync(db);
        var sources = new[]
        {
            WriteSourceFile("ok.jpg", MinimalJpeg()),
            WriteSourceFile("bad.jpg", MinimalJpeg()),
        };
        var before = sources.ToDictionary(s => s, Sha256Of);
        var api = new FakeOcr { Default = OcrOk("CHQ-SRC-001", "UNKNOWN-999", "100.00") };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            sources.Select(s => UploadOf(s, "image/jpeg")).ToList<IFormFile>(), _webRoot);
        await svc.ConfirmImportAsync(ToConfirm(preview), 1, _webRoot);

        foreach (var s in sources)
        {
            Assert.True(File.Exists(s));
            Assert.Equal(before[s], Sha256Of(s));
        }
    }

    // ---------- 6: cleanup never targets outside app-owned dirs ----------

    [Fact]
    public void CleanupGuard_RefusesForeignPaths()
    {
        var owned = Path.Combine(_webRoot, "uploads", "cheques", "batch-temp");
        Directory.CreateDirectory(owned);

        var ownedFile = Path.Combine(owned, "staged.jpg");
        File.WriteAllBytes(ownedFile, MinimalJpeg());
        var foreign = WriteSourceFile("external.jpg", MinimalJpeg());

        Assert.True(ChequeBatchImportService.IsOwnedPath(ownedFile, owned));
        Assert.False(ChequeBatchImportService.IsOwnedPath(foreign, owned));
        Assert.False(ChequeBatchImportService.IsOwnedPath(
            Path.Combine(_sourceDir, "..", Path.GetFileName(foreign)), owned));
        Assert.False(ChequeBatchImportService.IsOwnedPath(null, owned));
        Assert.False(ChequeBatchImportService.IsOwnedPath(ownedFile, null));
    }
}
