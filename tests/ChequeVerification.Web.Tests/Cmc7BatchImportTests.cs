using System.Text;
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
/// CMC7-authoritative batch import tests driven by the supervisor REAL dataset
/// (Phase 1 audit, scans 20260226, CDM rectos). Canonical values:
/// cheques 3177634/35/41/43/42/45/44, account 021780000000000000000085.
/// Raw CMC7 strings below are the EXACT OCR lines observed during the audit.
/// </summary>
public class Cmc7BatchImportTests : IDisposable
{
    public const string DemoAccount = "021780000000000000000085";

    private static readonly (string File, string Cheque, string Raw)[] Rectos =
    {
        ("20260226141441_0001+.jpg", "3177634", "306±3177634A021780A0000000000000000A85d"),
        ("20260226141441_0003+.jpg", "3177635", "306:31776354021780400000000000000004852"),
        ("20260226141441_0005+.jpg", "3177641", "30683177641A021780A0000000000000000A85d"),
        ("20260226141441_0007+.jpg", "3177643", "306±3177643A021780A0000000000000000A85d"),
        ("20260226141441_0009+.jpg", "3177642", "306±3177642A021780A0000000000000000A85"),
        ("20260226141441_0011+.jpg", "3177645", "306±3177645A021780A00000000000000004851"),
        // 0013: observed OCR line truncated the final "85" — parser rejects it.
        ("20260226141441_0013+.jpg", "3177644", "306±3177644A021780A0000000000000000"),
    };

    private readonly string _webRoot;

    public Cmc7BatchImportTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "cmc7-import-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_webRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { }
    }

    // ---------- helpers ----------

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

    private static ChequeOcrResponseDto Cmc7Ocr(
        string cheque7, string raw, string printedCheque,
        string? amountNumeric = "300.00", bool valid = true, string? crossCheck = null)
        => new()
        {
            Success = true,
            Message = "OCR effectuée avec succès.",
            FullText = $"CRÉDIT DU MAROC\n021 780 0000 000 000 00000 0 85\nTEL:022473887\n"
                + $"253 {cheque7}\nChèque N°\n{raw}",
            Lines = new() { new ChequeOcrLineDto { Text = raw, Confidence = 0.9 } },
            Fields = new ChequeOcrFieldsDto
            {
                ChequeNumber = "022473887", // generic extractor trap: agency TEL
                AccountNumber = null,
                AmountNumeric = amountNumeric,
                Cmc7Raw = raw,
                Cmc7ChequeNumber = valid ? cheque7 : null,
                Cmc7AccountNumber = valid ? DemoAccount : null,
                Cmc7Valid = valid,
                Cmc7Error = valid ? null : "CMC7 invalide (structure non conforme).",
                PrintedChequeNumber = "253" + printedCheque,
                PrintedAccountNumber = DemoAccount,
                Cmc7CrossCheck = crossCheck
                    ?? (valid ? "cheque-suffix-compatible;account-exact" : null),
            },
            ProcessingMs = 10,
            Lang = "fr",
            Device = "cpu"
        };

    private DbContextOptions<ChequeVerificationDbContext> Options()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"cmc7-db-{Guid.NewGuid():N}").Options;

    private static async Task SeedDemoCustomerAsync(ChequeVerificationDbContext db)
    {
        db.Customers.Add(new Customer
        {
            CustomerNumber = "CLI-CMC7-REAL",
            FullName = "Client CMC7 Réel",
            AccountNumber = DemoAccount,
            CreatedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();
    }

    private ChequeBatchImportService Service(ChequeVerificationDbContext db, IVerificationApiClient api)
        => new(db, api, NullLogger<ChequeBatchImportService>.Instance);

    private sealed class FakeCmc7Ocr : IVerificationApiClient
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

    // ---------- A: 6/7 valid rectos parse, same account, distinct cheques ----------

    [Fact]
    public async Task A_ValidRectos_ParseShareAccountDistinctCheques()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var api = new FakeCmc7Ocr();
        var files = new List<IFormFile>();
        foreach (var (file, cheque, raw) in Rectos.Take(6))
        {
            api.ByFile[file] = Cmc7Ocr(cheque, raw, cheque);
            files.Add(MakeFile(MinimalJpeg(), file, "image/jpeg"));
        }

        var preview = await Service(db, api).BuildPreviewAsync(files, _webRoot);

        Assert.Equal(6, preview.Rows.Count);
        Assert.All(preview.Rows, r => Assert.Equal(BatchImportSources.Cmc7, r.MappingSource));
        Assert.All(preview.Rows, r => Assert.Equal(BatchImportRowStatuses.Pret, r.Status));
        Assert.All(preview.Rows, r => Assert.Equal(DemoAccount, r.AccountNumber));
        Assert.All(preview.Rows, r => Assert.Equal(DemoAccount, r.Cmc7Account));
        var cheques = preview.Rows.Select(r => r.ChequeNumber).ToList();
        Assert.Equal(6, cheques.Distinct().Count());
        // Canonical 7-digit identity: no "253" prefix fabrication.
        Assert.All(preview.Rows, r => Assert.Matches(@"^\d{7}$", r.ChequeNumber ?? ""));
        Assert.All(preview.Rows, r => Assert.False((r.ChequeNumber ?? "").StartsWith("253")));
        // Printed body kept as diagnostic only.
        Assert.All(preview.Rows, r => Assert.Matches(@"^2533\d{6}$", r.PrintedCheque ?? ""));
    }

    // ---------- B: exact canonical cheque values ----------

    [Theory]
    [InlineData(0, "3177634")]
    [InlineData(1, "3177635")]
    [InlineData(2, "3177641")]
    [InlineData(3, "3177643")]
    [InlineData(4, "3177642")]
    [InlineData(5, "3177645")]
    public async Task B_CanonicalChequeNumbers_MatchDataset(int index, string expected)
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var (file, cheque, raw) = Rectos[index];
        var api = new FakeCmc7Ocr { Default = Cmc7Ocr(cheque, raw, cheque) };

        var preview = await Service(db, api).BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), file, "image/jpeg") }, _webRoot);

        var row = Assert.Single(preview.Rows);
        Assert.Equal(expected, row.ChequeNumber);
        Assert.Equal(expected, row.Cmc7Cheque);
        Assert.Equal(DemoAccount, row.AccountNumber);
    }

    // ---------- C: truncated 0013-style CMC7 rejected, body NOT substituted ----------

    [Fact]
    public async Task C_TruncatedCmc7_RejectedWithoutBodySubstitution()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var (file, cheque, raw) = Rectos[6]; // 0013 truncated line
        var api = new FakeCmc7Ocr { Default = Cmc7Ocr(cheque, raw, cheque, valid: false) };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), file, "image/jpeg") }, _webRoot);

        var row = Assert.Single(preview.Rows);
        Assert.False(row.IsInsertable);
        Assert.Equal(BatchImportRowStatuses.Cmc7Invalide, row.Status);
        // Body values visible as diagnostics but never promoted to identity.
        Assert.Equal("253" + cheque, row.PrintedCheque);
        Assert.NotEqual("253" + cheque, row.ChequeNumber);
        Assert.Null(row.ChequeNumber);

        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 1, _webRoot);
        Assert.Equal(0, result.ImportedCount);
        Assert.False(await db.Cheques.AnyAsync());
    }

    // ---------- D: missing CMC7 rejected despite plausible body/generic values ----------

    [Fact]
    public async Task D_MissingCmc7WithBodyValues_RejectedNoSubstitution()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var dto = new ChequeOcrResponseDto
        {
            Success = true,
            Message = "ok",
            FullText = "CRÉDIT DU MAROC\n021 780 0000 000 000 00000 0 85\nTEL:022473887\n253 3177634\nChèque N°",
            Lines = new(),
            Fields = new ChequeOcrFieldsDto
            {
                ChequeNumber = "022473887", // generic trap: must not become identity
                AccountNumber = null,
                AmountNumeric = "300.00",
                Cmc7Valid = false,
                PrintedChequeNumber = "2533177634",
                PrintedAccountNumber = DemoAccount,
            },
            ProcessingMs = 5,
            Lang = "fr",
            Device = "cpu"
        };
        var api = new FakeCmc7Ocr { Default = dto };

        var preview = await Service(db, api).BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), "x.jpg", "image/jpeg") }, _webRoot);

        var row = Assert.Single(preview.Rows);
        Assert.False(row.IsInsertable);
        Assert.Equal(BatchImportRowStatuses.Cmc7NonDetecte, row.Status);
        Assert.NotEqual("022473887", row.ChequeNumber);
        Assert.Null(row.ChequeNumber);
    }

    // ---------- E: exact customer mapping + import ----------

    [Fact]
    public async Task E_ValidCmc7_MapsToExactCustomerAndImports()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var expected = await db.Customers.SingleAsync(c => c.AccountNumber == DemoAccount);
        var (file, cheque, raw) = Rectos[0];
        var api = new FakeCmc7Ocr { Default = Cmc7Ocr(cheque, raw, cheque) };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), file, "image/jpeg") }, _webRoot);
        var row = Assert.Single(preview.Rows);
        Assert.Equal(expected.CustomerId, row.CustomerId);
        Assert.Equal("Client CMC7 Réel", row.CustomerFullName);

        var result = await svc.ConfirmImportAsync(ToConfirm(preview), 7, _webRoot);
        Assert.Equal(1, result.ImportedCount);
        var stored = await db.Cheques.Include(c => c.Customer).SingleAsync();
        Assert.Equal("3177634", stored.ChequeNumber);
        Assert.Equal(expected.CustomerId, stored.CustomerId);
        Assert.Equal(DemoAccount, stored.Customer.AccountNumber);
        Assert.Equal((byte)1, stored.Status); // En attente
    }

    // ---------- F: duplicate canonical CMC7 number rejected ----------

    [Fact]
    public async Task F_DuplicateCmc7Number_Rejected()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var cust = await db.Customers.SingleAsync(c => c.AccountNumber == DemoAccount);
        db.Cheques.Add(new Cheque
        {
            CustomerId = cust.CustomerId, ImportedByUserId = 1, ChequeNumber = "3177634",
            Amount = 300, ImagePath = "/uploads/cheques/x.png", Status = 1, UploadedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();

        var (file, cheque, raw) = Rectos[0];
        var api = new FakeCmc7Ocr { Default = Cmc7Ocr(cheque, raw, cheque) };
        var preview = await Service(db, api).BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), file, "image/jpeg") }, _webRoot);

        Assert.Equal(BatchImportRowStatuses.Duplique, Assert.Single(preview.Rows).Status);
    }

    // ---------- G: verso not importable ----------

    [Fact]
    public async Task G_Verso_NotImportable()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var dto = new ChequeOcrResponseDto
        {
            Success = true,
            Message = "ok",
            FullText = "AVIS IMPORTANT\nTRG3 - 1025 - 0100 - 07\nloi 15-95 Code de Commerce",
            Lines = new() { new ChequeOcrLineDto { Text = "AVIS IMPORTANT", Confidence = 0.99 } },
            Fields = new ChequeOcrFieldsDto { IsProbableVerso = true },
            ProcessingMs = 5,
            Lang = "fr",
            Device = "cpu"
        };
        var api = new FakeCmc7Ocr { Default = dto };

        var preview = await Service(db, api).BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), "verso.jpg", "image/jpeg") }, _webRoot);

        var row = Assert.Single(preview.Rows);
        Assert.False(row.IsInsertable);
        Assert.Equal(BatchImportRowStatuses.VersoNonImportable, row.Status);
    }

    // ---------- H: body mismatch with valid CMC7 blocked ----------

    [Fact]
    public async Task H_BodyMismatchWithValidCmc7_Blocked()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var (file, cheque, raw) = Rectos[0];
        var dto = Cmc7Ocr(cheque, raw, "3999999", crossCheck: "CHEQUE-MISMATCH;account-exact");
        dto.Fields!.PrintedChequeNumber = "2533999999";
        var api = new FakeCmc7Ocr { Default = dto };

        var preview = await Service(db, api).BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), file, "image/jpeg") }, _webRoot);

        var row = Assert.Single(preview.Rows);
        Assert.False(row.IsInsertable);
        Assert.Equal(BatchImportRowStatuses.IncoherenceDonnees, row.Status);
        // CMC7 identity preserved, body value not promoted.
        Assert.Equal("3177634", row.ChequeNumber);
    }

    // ---------- I: mixed batch partial success ----------

    [Fact]
    public async Task I_MixedBatch_PartialSuccessPreserved()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var api = new FakeCmc7Ocr();
        api.ByFile["ok1.jpg"] = Cmc7Ocr("3177634", Rectos[0].Raw, "3177634", "100.00");
        api.ByFile["ok2.jpg"] = Cmc7Ocr("3177635", Rectos[1].Raw, "3177635", "200.00");
        api.ByFile["bad.jpg"] = Cmc7Ocr("3177641", Rectos[2].Raw, "3177641", "300.00", valid: false);
        api.ByFile["dup.jpg"] = Cmc7Ocr("3177634", Rectos[0].Raw, "3177634", "400.00"); // in-batch duplicate

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

    // ---------- J: confirm revalidation rejects tampered identity ----------

    [Fact]
    public async Task J_ConfirmRevalidation_RejectsTamperedIdentity()
    {
        await using var db = new ChequeVerificationDbContext(Options());
        await SeedDemoCustomerAsync(db);
        var (file, cheque, raw) = Rectos[0];
        var api = new FakeCmc7Ocr { Default = Cmc7Ocr(cheque, raw, cheque) };

        var svc = Service(db, api);
        var preview = await svc.BuildPreviewAsync(
            new List<IFormFile> { MakeFile(MinimalJpeg(), file, "image/jpeg") }, _webRoot);

        var confirm = ToConfirm(preview);
        confirm.Rows[0].ChequeNumber = "9999999"; // tampered hidden value
        var result = await svc.ConfirmImportAsync(confirm, 1, _webRoot);

        Assert.Equal(0, result.ImportedCount);
        Assert.False(await db.Cheques.AnyAsync());
    }

    // ---------- K: structural shape gates ----------

    [Theory]
    [InlineData("3177634", true)]
    [InlineData("317763", false)]
    [InlineData("31776344", false)]
    [InlineData("2533177634", false)] // body form must not pass as CMC7 identity
    [InlineData("022473887", false)]  // phone trap
    [InlineData(null, false)]
    public void K_Cmc7ChequeShape_GatesCorrectly(string? value, bool expected)
        => Assert.Equal(expected, ChequeBatchImportService.IsCmc7ChequeShape(value));

    [Theory]
    [InlineData("021780000000000000000085", true)]
    [InlineData("02178000000000000000008", false)]
    [InlineData("0217800000000000000000855", false)]
    [InlineData("021 780 0000 000 000 00000 0 85", false)] // unnormalized never passes
    [InlineData(null, false)]
    public void L_Cmc7AccountShape_GatesCorrectly(string? value, bool expected)
        => Assert.Equal(expected, ChequeBatchImportService.IsCmc7AccountShape(value));
}
