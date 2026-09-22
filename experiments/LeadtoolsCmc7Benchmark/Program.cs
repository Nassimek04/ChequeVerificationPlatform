using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Leadtools;
using Leadtools.Codecs;
using Leadtools.Forms.Commands;
using Leadtools.ImageProcessing.Core;
using Leadtools.Ocr;

// ============================================================================
// LEADTOOLS CMC7/MICR BENCHMARK (isolated experiment - NOT production code)
// Dataset: C:\Dev\fwchq (7 rectos + 7 versos)
// API under test: CMC7CodeDetectionCommand + BankCheckReader (official LEADTOOLS)
// Ground truth is used ONLY for post-hoc validation, never to correct output.
// ============================================================================

const string DatasetDir = @"C:\Dev\fwchq";
const string OutputDir = @"C:\Dev\ChequeVerificationPlatform\experiments\LeadtoolsCmc7Benchmark";

string[] rectos = [
    "20260226141441_0001+.jpg",
    "20260226141441_0003+.jpg",
    "20260226141441_0005+.jpg",
    "20260226141441_0007+.jpg",
    "20260226141441_0009+.jpg",
    "20260226141441_0011+.jpg",
    "20260226141441_0013+.jpg",
];

var expectedCheques = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
{
    ["20260226141441_0001+.jpg"] = "3177634",
    ["20260226141441_0003+.jpg"] = "3177635",
    ["20260226141441_0005+.jpg"] = "3177641",
    ["20260226141441_0007+.jpg"] = "3177643",
    ["20260226141441_0009+.jpg"] = "3177642",
    ["20260226141441_0011+.jpg"] = "3177645",
    ["20260226141441_0013+.jpg"] = "3177644",
};
const string ExpectedAccount = "021780000000000000000085";
string[] versos = ["20260226141441_0002+.jpg"];

Console.WriteLine("=== LEADTOOLS CMC7 benchmark (isolated experiment) ===");
Console.WriteLine($"SDK packages: Leadtools.Document.Sdk 23.0.0.7 + Leadtools.Formats.Raster.Common 23.0.0.7");
Console.WriteLine($"Dataset: {DatasetDir}");
Console.WriteLine();

// ---- 0. License probe (do NOT bypass licensing) ----
string licFile = Path.Combine(OutputDir, "Leadtools.lic");
string keyFile = Path.Combine(OutputDir, "Leadtools.lic.key");
Console.WriteLine($"License probe: looking for {licFile} (+ .key)");
Console.WriteLine($"  Leadtools.lic present: {File.Exists(licFile)}");
Console.WriteLine($"  Leadtools.lic.key present: {File.Exists(keyFile)}");
bool licenseSet = false;
string licenseError = "";
try
{
    if (File.Exists(licFile) && File.Exists(keyFile))
    {
        byte[] licBytes = File.ReadAllBytes(licFile);
        string keyText = File.ReadAllText(keyFile);
        RasterSupport.SetLicense(licBytes, keyText);
        licenseSet = true;
        Console.WriteLine("  RasterSupport.SetLicense: OK");
    }
    else
    {
        licenseError = "No LEADTOOLS.LIC / .LIC.key files provided. " +
            "Evaluation license requires free registration + Full Evaluation download " +
            "(https://www.leadtools.com/downloads) which emails a 60-day LIC+key. " +
            "Not bypassed; proceeding to document the exact runtime failure.";
        Console.WriteLine($"  NO LICENSE: {licenseError}");
    }
}
catch (Exception ex)
{
    licenseError = $"SetLicense threw {ex.GetType().Name}: {ex.Message}";
    Console.WriteLine($"  SetLicense FAILED: {licenseError}");
}
Console.WriteLine($"  Kernel version: {typeof(RasterImage).Assembly.GetName().Version}");
Console.WriteLine();

// ---- 1. OCR runtime probe ----
// Official tutorial requires OcrLEADRuntime dir from the FULL SDK install:
//   <INSTALL_DIR>\LEADTOOLS23\Bin\Common\OcrLEADRuntime
string[] runtimeCandidates =
[
    @"C:\LEADTOOLS23\Bin\Common\OcrLEADRuntime",
    Path.Combine(OutputDir, "OcrLEADRuntime"),
];
string? ocrRuntime = runtimeCandidates.FirstOrDefault(Directory.Exists);
Console.WriteLine($"OCR runtime probe (OcrLEADRuntime from full SDK install):");
foreach (var c in runtimeCandidates) Console.WriteLine($"  {c} exists={Directory.Exists(c)}");
Console.WriteLine($"  Selected: {(ocrRuntime ?? "<NONE - full SDK not installed>")}");
Console.WriteLine();

var results = new List<Row>();

if (!licenseSet || ocrRuntime is null)
{
    // Record blocked rows for all files so the report table is explicit.
    foreach (var f in rectos.Concat(versos))
    {
        results.Add(new Row
        {
            File = f,
            RawMicrOutput = "",
            NormalizedDigits = "",
            DetectedCheque = "",
            DetectedAccount = "",
            ElapsedMs = 0,
            Success = false,
            Error = licenseSet
                ? "BLOCKED: OcrLEADRuntime missing (requires LEADTOOLS full evaluation install)."
                : $"BLOCKED: {licenseError}",
        });
    }
}
else
{
    using var codecs = new RasterCodecs();
    IOcrEngine? ocrEngine = null;
    try
    {
        ocrEngine = OcrEngineManager.CreateEngine(OcrEngineType.LEAD);
        ocrEngine.Startup(codecs, null, null, ocrRuntime);
        Console.WriteLine("OCR engine startup: OK");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"OCR engine startup FAILED: {ex.GetType().Name}: {ex.Message}");
    }
    Console.WriteLine();

    if (ocrEngine is null)
    {
        foreach (var f in rectos.Concat(versos))
            results.Add(new Row { File = f, ElapsedMs = 0, Success = false, RawMicrOutput = "", NormalizedDigits = "", DetectedCheque = "", DetectedAccount = "", Error = "BLOCKED: OCR engine startup failed." });
    }
    else
    {
        foreach (var f in rectos.Concat(versos))
            results.Add(ProcessOne(codecs, ocrEngine, Path.Combine(DatasetDir, f)));
        ocrEngine.Shutdown();
    }
}

// ---- 2. Post-hoc validation (ground truth NEVER feeds recognition) ----
int chequeOk = 0, accountOk = 0, fullyOk = 0;
foreach (var r in results.Where(r => rectos.Contains(r.File)))
{
    r.ExpectedCheque = expectedCheques[r.File];
    r.ExpectedAccount = ExpectedAccount;
    r.ChequeMatch = string.Equals(r.DetectedCheque, r.ExpectedCheque, StringComparison.Ordinal);
    r.AccountMatch = string.Equals(r.DetectedAccount, ExpectedAccount, StringComparison.Ordinal);
    if (r.ChequeMatch) chequeOk++;
    if (r.AccountMatch) accountOk++;
    if (r.ChequeMatch && r.AccountMatch && r.Success) fullyOk++;
}

// ---- 3. Console table ----
Console.WriteLine("File | Raw | Normalized | Cheque(det/exp) | AccountMatch | ms | OK | Error");
foreach (var r in results)
    Console.WriteLine($"{r.File} | raw='{r.RawMicrOutput}' | norm='{r.NormalizedDigits}' | cheque='{r.DetectedCheque}'/exp='{r.ExpectedCheque}' match={r.ChequeMatch} | acctMatch={r.AccountMatch} | {r.ElapsedMs}ms | success={r.Success} | {r.Error}");

Console.WriteLine();
Console.WriteLine($"Cheque recognition accuracy: {chequeOk} / 7");
Console.WriteLine($"Account recognition accuracy: {accountOk} / 7");
Console.WriteLine($"Fully correct cheques: {fullyOk} / 7");
var r0013 = results.FirstOrDefault(r => r.File.Contains("_0013+"));
Console.WriteLine($"0013+ (PaddleOCR strict-fail case): detected='{r0013?.DetectedCheque}' expected='3177644' match={r0013?.ChequeMatch} success={r0013?.Success} err={r0013?.Error}");
var verso = results.FirstOrDefault(r => r.File.Contains("_0002+"));
Console.WriteLine($"Verso negative test ({verso?.File}): success={verso?.Success} cheque='{verso?.DetectedCheque}' acct='{verso?.DetectedAccount}' (expected: no valid CMC7) err={verso?.Error}");
double avg = results.Where(r => r.Success).Select(r => (double)r.ElapsedMs).DefaultIfEmpty(0).Average();
Console.WriteLine($"Average processing time (successful only): {avg:F0} ms");

// ---- 4. Artifacts ----
string jsonPath = Path.Combine(OutputDir, "benchmark_results.json");
var payload = new
{
    sdk = "LEADTOOLS v23 (Leadtools.Document.Sdk 23.0.0.7, Leadtools.Formats.Raster.Common 23.0.0.7, nuget.org)",
    api = "CMC7CodeDetectionCommand + MICRCodeDetectionCommand + BankCheckReader(MicrFontType.Cmc7) + IOcrEngine(LEAD)",
    licenseSet,
    licenseError,
    ocrRuntime,
    productionUntouched = true,
    chequeAccuracy = $"{chequeOk}/7",
    accountAccuracy = $"{accountOk}/7",
    fullyCorrect = $"{fullyOk}/7",
    avgMsSuccessful = avg,
    rows = results,
};
File.WriteAllText(jsonPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Wrote {jsonPath}");

// ---- helpers ----
static Row ProcessOne(RasterCodecs codecs, IOcrEngine ocrEngine, string path)
{
    var row = new Row { File = Path.GetFileName(path) };
    var sw = Stopwatch.StartNew();
    try
    {
        using RasterImage image = codecs.Load(path);
        var micrReader = new BankCheckReader { OcrEngine = ocrEngine };

        var e13bCmd = new MICRCodeDetectionCommand { SearchingZone = new LeadRect(0, 0, image.Width, image.Height) };
        e13bCmd.Run(image);
        var cmc7Cmd = new CMC7CodeDetectionCommand();
        cmc7Cmd.Run(image);

        bool hasE13b = e13bCmd.MICRZone != LeadRect.Empty;
        bool hasCmc7 = cmc7Cmd.CMC7Zone != LeadRect.Empty;
        row.DetectionInfo = $"E13bZone={e13bCmd.MICRZone} CMC7Zone={cmc7Cmd.CMC7Zone}";

        if (hasCmc7)
            micrReader.MicrFontType = BankCheckMicrFontType.Cmc7;
        else if (hasE13b)
            micrReader.MicrFontType = BankCheckMicrFontType.E13b;
        else
        {
            row.Success = false;
            row.Error = "No MICR/CMC7 zone detected.";
            return row;
        }

        micrReader.ProcessImage(image);
        var sb = new StringBuilder();
        foreach (var kv in micrReader.Results)
        {
            if (kv.Key == "Signature") continue;
            sb.Append($"{kv.Key}={kv.Value?.Text};");
        }
        row.RawMicrOutput = sb.ToString();
        row.NormalizedDigits = Regex.Replace(row.RawMicrOutput ?? "", @"\D", "");
        // Parse fields WITHOUT ground-truth correction: report what the reader returned.
        row.DetectedCheque = micrReader.Results.TryGetValue("Check Number", out var cn) ? (cn?.Text?.Trim() ?? "") : "";
        if (string.IsNullOrEmpty(row.DetectedCheque) && micrReader.Results.TryGetValue("Auxiliary", out var aux))
            row.DetectedCheque = aux?.Text?.Trim() ?? "";
        row.DetectedAccount = micrReader.Results.TryGetValue("Account Number", out var an) ? (an?.Text?.Trim() ?? "") : "";
        row.Success = !string.IsNullOrWhiteSpace(row.RawMicrOutput);
        if (!row.Success) row.Error = "BankCheckReader returned empty results.";
    }
    catch (Exception ex)
    {
        row.Success = false;
        row.Error = $"{ex.GetType().Name}: {ex.Message}";
    }
    finally { sw.Stop(); row.ElapsedMs = sw.ElapsedMilliseconds; }
    return row;
}

sealed class Row
{
    public string File { get; set; } = "";
    public string RawMicrOutput { get; set; } = "";
    public string NormalizedDigits { get; set; } = "";
    public string DetectedCheque { get; set; } = "";
    public string DetectedAccount { get; set; } = "";
    public string ExpectedCheque { get; set; } = "";
    public string ExpectedAccount { get; set; } = "";
    public bool ChequeMatch { get; set; }
    public bool AccountMatch { get; set; }
    public long ElapsedMs { get; set; }
    public bool Success { get; set; }
    public string Error { get; set; } = "";
    public string DetectionInfo { get; set; } = "";
}
