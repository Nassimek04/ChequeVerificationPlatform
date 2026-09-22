using System.Text;
using ChequeVerification.Web.Controllers;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Cheques;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

/// <summary>
/// Phase 1 batch import tests (A–M): preview generation, exact account mapping,
/// duplicate handling, amount parsing, partial success, file rejection,
/// Status = En attente, customer linkage, and Controller denial.
/// </summary>
public class BatchChequeImportTests : IDisposable
{
    private readonly string _webRoot;

    public BatchChequeImportTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "batch-import-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_webRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { }
    }

    // ---------- helpers ----------

    private static byte[] MinimalPng(int width = 100, int height = 100)
    {
        var bytes = new List<byte> { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
        bytes.AddRange(new byte[] { 0, 0, 0, 13 }); // IHDR length
        bytes.AddRange(Encoding.ASCII.GetBytes("IHDR"));
        bytes.Add((byte)(width >> 24)); bytes.Add((byte)(width >> 16)); bytes.Add((byte)(width >> 8)); bytes.Add((byte)width);
        bytes.Add((byte)(height >> 24)); bytes.Add((byte)(height >> 16)); bytes.Add((byte)(height >> 8)); bytes.Add((byte)height);
        bytes.AddRange(new byte[] { 8, 2, 0, 0, 0 }); // bit depth / color / etc.
        bytes.AddRange(new byte[100]); // filler
        return bytes.ToArray();
    }

    private static byte[] MinimalJpeg()
    {
        var bytes = new List<byte> { 0xFF, 0xD8, 0xFF, 0xE0 };
        bytes.AddRange(new byte[200]);
        return bytes.ToArray();
    }

    private static FormFile MakeFile(byte[] bytes, string fileName, string contentType)
        => new(new MemoryStream(bytes), 0, bytes.Length, "Files", fileName)
        {
            Headers = new HeaderDictionary(),
            ContentType = contentType
        };

    private static ChequeOcrResponseDto OcrOk(string cheque, string? account, string? amountNumeric, string fullText = "")
        => new()
        {
            Success = true,
            Message = "OCR effectuée avec succès.",
            FullText = fullText,
            Lines = new(),
            Fields = new ChequeOcrFieldsDto
            {
                ChequeNumber = cheque,
                AccountNumber = account,
                AmountNumeric = amountNumeric
            },
            ProcessingMs = 10,
            Lang = "fr",
            Device = "cpu"
        };

    private DbContextOptions<ChequeVerificationDbContext> Options()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"batch-db-{Guid.NewGuid():N}").Options;

    private async Task SeedCustomersAsync(ChequeVerificationDbContext db)
    {
        for (var i = 1; i <= 5; i++)
        {
            db.Customers.Add(new Customer
            {
                CustomerNumber = $"CUST-BLIND-{i}",
                FullName = i == 1 ? "Blind Test Customer" : $"Blind Customer {i}",
                AccountNumber = $"BLIND-000{i}",
                CreatedAt = DateTime.UtcNow
            });
        }
        await db.SaveChangesAsync();
    }

    private ChequeBatchImportService Service(ChequeVerificationDbContext db, IVerificationApiClient api)
        => new(db, api, NullLogger<ChequeBatchImportService>.Instance);

    private sealed class FakeBatchOcr : IVerificationApiClient
    {
        public Dictionary<string, ChequeOcrResponseDto?> ByFile { get; } = new(StringComparer.OrdinalIgnoreCase);
        public ChequeOcrResponseDto? Default { get; set; }
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default) => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<ImageAnalysisResponseDto?>(null);
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureExtractionResponseDto?>(null);
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureDebugResponseDto?>(null);
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default) => Task.FromResult<SignatureComparisonResponseDto?>(null);
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default) => Task.FromResult<SignatureAiComparisonResponseDto?>(null);
        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult(ByFile.TryGetValue(f, out var r) ? r : Default);
    }

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

    // ---------- A: multiple valid images => preview rows ----------

    [Fact]
    public async Task A_MultipleValidImages_ProducePreviewRows()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var api = new FakeBatchOcr();
        api.ByFile["CHQ-BATCH-001.jpg"] = OcrOk("CHQ-BATCH-001", "BLIND-0001", "12500.00");
        api.ByFile["CHQ-BATCH-002.jpg"] = OcrOk("CHQ-BATCH-002", "BLIND-0002", "12500.00");
        api.ByFile["CHQ-BATCH-003.png"] = OcrOk("CHQ-BATCH-003", "BLIND-0003", "12500.00");

        var files = new List<IFormFile>
        {
            MakeFile(MinimalJpeg(), "CHQ-BATCH-001.jpg", "image/jpeg"),
            MakeFile(MinimalJpeg(), "CHQ-BATCH-002.jpg", "image/jpeg"),
            MakeFile(MinimalPng(), "CHQ-BATCH-003.png", "image/png"),
        };

        var preview = await Service(db, api).BuildPreviewAsync(files, _webRoot);

        Assert.Equal(3, preview.Rows.Count);
        Assert.All(preview.Rows, r => Assert.True(r.IsInsertable));
        Assert.All(preview.Rows, r => Assert.Equal(BatchImportRowStatuses.Pret, r.Status));
    }

    // ---------- B: BLIND-0001 maps to correct customer ----------

    [Fact]
    public async Task B_BlindAccount_MapsToCorrectCustomer()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var expected = await db.Customers.SingleAsync(c => c.AccountNumber == "BLIND-0001");
        var api = new FakeBatchOcr
        {
            Default = OcrOk("CHQ-BATCH-001", " blind - 0001 ", "12500.00")
        };

        var preview = await Service(db, api).BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), "c.jpg", "image/jpeg") }, _webRoot);

        var row = Assert.Single(preview.Rows);
        Assert.Equal("BLIND-0001", row.AccountNumber);
        Assert.Equal(expected.CustomerId, row.CustomerId);
        Assert.Equal("Blind Test Customer", row.CustomerFullName);
    }

    // ---------- C: unknown account => unresolved, no insert ----------

    [Fact]
    public async Task C_UnknownAccount_UnresolvedAndNotInserted()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var api = new FakeBatchOcr { Default = OcrOk("CHQ-BATCH-001", "UNKNOWN-999", "100.00") };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), "c.jpg", "image/jpeg") }, _webRoot);
        var row = Assert.Single(preview.Rows);
        Assert.False(row.IsInsertable);
        Assert.Equal(BatchImportRowStatuses.ClientIntrouvable, row.Status);

        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 1, _webRoot);
        Assert.Equal(0, result.ImportedCount);
        Assert.False(await db.Cheques.AnyAsync());
    }

    // ---------- D: missing account => no insert ----------

    [Fact]
    public async Task D_MissingAccount_NoInsert()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var api = new FakeBatchOcr { Default = OcrOk("CHQ-BATCH-001", null, "100.00") };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), "c.jpg", "image/jpeg") }, _webRoot);
        var row = Assert.Single(preview.Rows);
        Assert.False(row.IsInsertable);
        Assert.Equal(BatchImportRowStatuses.CompteNonDetecte, row.Status);

        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 1, _webRoot);
        Assert.False(await db.Cheques.AnyAsync());
    }

    // ---------- E: duplicate cheque number => no duplicate ----------

    [Fact]
    public async Task E_DuplicateChequeNumber_NoSecondRecord()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var cust = await db.Customers.SingleAsync(c => c.AccountNumber == "BLIND-0001");
        db.Cheques.Add(new Cheque
        {
            CustomerId = cust.CustomerId, ImportedByUserId = 1, ChequeNumber = "CHQ-BATCH-001",
            Amount = 10, ImagePath = "/uploads/cheques/x.png", Status = 1, UploadedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();

        var api = new FakeBatchOcr { Default = OcrOk("CHQ-BATCH-001", "BLIND-0001", "100.00") };
        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), "c.jpg", "image/jpeg") }, _webRoot);
        Assert.Equal(BatchImportRowStatuses.Duplique, Assert.Single(preview.Rows).Status);

        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 1, _webRoot);
        Assert.Equal(0, result.ImportedCount);
        Assert.Equal(1, await db.Cheques.CountAsync(c => c.ChequeNumber == "CHQ-BATCH-001"));
    }

    // ---------- F: amount parsing ----------

    [Theory]
    [InlineData("12 500,00 MAD", 12500.00)]
    [InlineData("12.500,00", 12500.00)]
    [InlineData("12500.00", 12500.00)]
    [InlineData("12 500 MAD", 12500.00)]
    [InlineData("40.000,00", 40000.00)]
    [InlineData("12 500,00 MAD", 12500.00)] // NBSP group separator + currency
    public void F_AmountParsing_ReturnsCorrectDecimal(string raw, decimal expected)
    {
        Assert.True(ChequeBatchImportService.TryParseAmount(raw, out var amount));
        Assert.Equal(expected, amount);
    }

    [Fact]
    public void F_AmountParsing_HandlesUnicodeSpaces()
    {
        // NBSP and narrow NBSP (fr-FR ICU grouping) built from char codes
        // so the source stays ASCII-only.
        var nbsp = "12" + (char)0x00A0 + "500,00";
        var narrow = "12" + (char)0x202F + "500,00";
        Assert.True(ChequeBatchImportService.TryParseAmount(nbsp, out var a1));
        Assert.Equal(12500.00m, a1);
        Assert.True(ChequeBatchImportService.TryParseAmount(narrow, out var a2));
        Assert.Equal(12500.00m, a2);
    }

    // ---------- G: invalid amount => correction required / blocked ----------

    [Fact]
    public async Task G_InvalidAmount_BlockedUntilCorrected()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var api = new FakeBatchOcr { Default = OcrOk("CHQ-BATCH-001", "BLIND-0001", "montant illisible") };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), "c.jpg", "image/jpeg") }, _webRoot);
        var row = Assert.Single(preview.Rows);
        Assert.Equal(BatchImportRowStatuses.MontantAVerifier, row.Status);
        Assert.False(row.IsInsertable);

        // Confirm without correction stays rejected.
        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 1, _webRoot);
        Assert.Equal(0, result.ImportedCount);
        Assert.False(await db.Cheques.AnyAsync());

        // With a valid correction the same row imports.
        var confirm = ToConfirm(preview);
        confirm.Rows[0].CorrectedAmount = "12 500,00 MAD";
        var result2 = await svc.ConfirmImportAsync(confirm, 1, _webRoot);
        Assert.Equal(1, result2.ImportedCount);
        Assert.Equal(12500.00m, (await db.Cheques.SingleAsync()).Amount);
    }

    // ---------- H: mixed batch => partial success ----------

    [Fact]
    public async Task H_MixedBatch_ValidImportInvalidRejected()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var api = new FakeBatchOcr();
        api.ByFile["ok1.jpg"] = OcrOk("CHQ-BATCH-001", "BLIND-0001", "100.00");
        api.ByFile["ok2.jpg"] = OcrOk("CHQ-BATCH-002", "BLIND-0002", "200.00");
        api.ByFile["bad.jpg"] = OcrOk("CHQ-BATCH-003", "UNKNOWN-1", "300.00");
        api.ByFile["dup.jpg"] = OcrOk("CHQ-BATCH-001", "BLIND-0003", "400.00"); // within-batch duplicate

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(new List<IFormFile>
        {
            MakeFile(MinimalJpeg(), "ok1.jpg", "image/jpeg"),
            MakeFile(MinimalJpeg(), "ok2.jpg", "image/jpeg"),
            MakeFile(MinimalJpeg(), "bad.jpg", "image/jpeg"),
            MakeFile(MinimalJpeg(), "dup.jpg", "image/jpeg"),
        }, _webRoot);

        Assert.Equal(2, preview.ReadyCount);
        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 1, _webRoot);
        Assert.Equal(2, result.ImportedCount);
        Assert.Equal(2, result.IgnoredCount);
        Assert.Equal(2, await db.Cheques.CountAsync());
    }

    // ---------- I: unsupported type rejected ----------

    [Fact]
    public async Task I_UnsupportedFileType_Rejected()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var api = new FakeBatchOcr { Default = OcrOk("CHQ-BATCH-001", "BLIND-0001", "100.00") };

        var pdf = MakeFile(MinimalJpeg(), "doc.pdf", "application/pdf");
        var preview = await Service(db, api).BuildPreviewAsync(new List<IFormFile> { pdf }, _webRoot);

        var row = Assert.Single(preview.Rows);
        Assert.False(row.IsInsertable);
        Assert.Equal(BatchImportRowStatuses.ImageInvalide, row.Status);
    }

    // ---------- J: corrupted image rejected ----------

    [Fact]
    public async Task J_CorruptedImage_Rejected()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var api = new FakeBatchOcr { Default = OcrOk("CHQ-BATCH-001", "BLIND-0001", "100.00") };

        var corrupt = MakeFile(Encoding.ASCII.GetBytes("not an image at all ............"), "c.png", "image/png");
        var preview = await Service(db, api).BuildPreviewAsync(new List<IFormFile> { corrupt }, _webRoot);

        var row = Assert.Single(preview.Rows);
        Assert.False(row.IsInsertable);
        Assert.Equal(BatchImportRowStatuses.ImageInvalide, row.Status);
        Assert.Equal(0, await db.Cheques.CountAsync());
    }

    // ---------- K: status En attente ----------

    [Fact]
    public async Task K_ImportedCheque_StatusEnAttente()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var api = new FakeBatchOcr { Default = OcrOk("CHQ-BATCH-001", "BLIND-0001", "12500.00") };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), "c.jpg", "image/jpeg") }, _webRoot);
        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 7, _webRoot);

        Assert.Equal(1, result.ImportedCount);
        var cheque = await db.Cheques.SingleAsync();
        Assert.Equal((byte)1, cheque.Status);
    }

    // ---------- L: appears under correct customer ----------

    [Fact]
    public async Task L_ImportedCheque_LinkedToCorrectCustomer()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedCustomersAsync(db);
        var expected = await db.Customers.SingleAsync(c => c.AccountNumber == "BLIND-0002");
        var api = new FakeBatchOcr { Default = OcrOk("CHQ-BATCH-002", "BLIND-0002", "200.00") };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), "c.jpg", "image/jpeg") }, _webRoot);
        await svc.ConfirmImportAsync(ToConfirm(preview), 7, _webRoot);

        var cheque = await db.Cheques.Include(c => c.Customer).SingleAsync();
        Assert.Equal(expected.CustomerId, cheque.CustomerId);
        Assert.Equal("BLIND-0002", cheque.Customer.AccountNumber);
    }

    // ---------- M: Controller cannot access batch import ----------

    [Fact]
    public void M_ConfirmBatchImport_RequiresUtilAdminAntiforgeryAndDeniesControleur()
    {
        var method = typeof(ChequesController).GetMethod("ConfirmBatchImport");
        Assert.NotNull(method);
        Assert.NotNull(method!.GetCustomAttributes(typeof(HttpPostAttribute), true).FirstOrDefault());
        Assert.NotNull(method.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), true).FirstOrDefault());
        var auth = method.GetCustomAttributes(typeof(AuthorizeAttribute), true)
            .Cast<AuthorizeAttribute>().FirstOrDefault();
        Assert.NotNull(auth);
        var roles = (auth!.Roles ?? string.Empty).Split(',').Select(r => r.Trim()).ToHashSet();
        Assert.Contains("Utilisateur", roles);
        Assert.Contains("Administrateur", roles);
        Assert.DoesNotContain("Contrôleur", roles);
    }

    [Theory]
    [InlineData("GET")]
    [InlineData("POST")]
    public void M_BatchImportActions_DenyControleur(string verb)
    {
        var method = typeof(ChequesController).GetMethods()
            .Single(m => m.Name == "BatchImport"
                && (verb == "GET"
                    ? m.GetCustomAttributes(typeof(HttpGetAttribute), true).Any()
                    : m.GetCustomAttributes(typeof(HttpPostAttribute), true).Any()));
        if (verb == "POST")
        {
            Assert.NotNull(method.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), true)
                .FirstOrDefault());
        }
        var auth = method.GetCustomAttributes(typeof(AuthorizeAttribute), true)
            .Cast<AuthorizeAttribute>().FirstOrDefault();
        Assert.NotNull(auth);
        var roles = (auth!.Roles ?? string.Empty).Split(',').Select(r => r.Trim()).ToHashSet();
        Assert.Contains("Utilisateur", roles);
        Assert.Contains("Administrateur", roles);
        Assert.DoesNotContain("Contrôleur", roles);
    }
}
