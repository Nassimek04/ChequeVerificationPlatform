using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Verifications;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;
using System.Text.Json;

namespace ChequeVerification.Web.Tests;

/// <summary>
/// Hybrid localization (ROI first, full-document fallback):
/// Bilan mapping of the localization evidence + guard that V5-A is never
/// called when extraction fails.
/// </summary>
public class HybridExtractionLocalizationTests : IDisposable
{
    private readonly string _webRoot;
    private readonly FakeHybridApiClient _api = new();

    public HybridExtractionLocalizationTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "verify-hybrid-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_webRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { }
    }

    [Fact]
    public async Task DebugMapping_FallbackMode_MapsLocalizationEvidence()
    {
        await using var db = CreateDbContext();
        await SeedChequeAsync(db);
        _api.OnDebug = () => new SignatureDebugResponseDto
        {
            Success = true,
            ExtractionPipelineVersion = "2.4",
            OriginalWidth = 800,
            OriginalHeight = 355,
            CandidateRoi = new BoundingBoxDto { X = 0, Y = 0, Width = 800, Height = 355 },
            SignatureBbox = new BoundingBoxDto { X = 60, Y = 36, Width = 81, Height = 49 },
            ExtractionQuality = 0.7958,
            LocalizationMode = "global_fallback",
            LocalizationLabel = "Recherche globale de secours",
            RoiFailureReason = "aucun contenu détecté (bbox None).",
            FallbackCandidateCount = 2,
            FallbackSelectedIndex = 0,
            FallbackSelectedScore = 0.00193,
            FallbackSelectionReason = "Groupe 0 sélectionné",
            FinalCropWidth = 93,
            FinalCropHeight = 57,
            FallbackRejectedCandidates = new List<FallbackRejectedCandidateDto>
            {
                new() { Index = 1, Reason = "Rejeté : bloc trop dense.", Bbox = new BoundingBoxDto { X = 0, Y = 0, Width = 800, Height = 61 } },
            },
            FallbackCandidatesImageBase64 = "FFF=",
        };
        var svc = CreateService(db);

        var diag = await svc.GetExtractionDiagnosticAsync(1);

        Assert.True(diag.Success);
        Assert.Equal("global_fallback", diag.LocalizationMode);
        Assert.Equal("Recherche globale automatique", diag.LocalizationDisplay);
        Assert.True(diag.IsGlobalFallback);
        Assert.Equal(2, diag.FallbackCandidateCount);
        Assert.Equal(0, diag.FallbackSelectedIndex);
        Assert.Equal(0.00193, diag.FallbackSelectedScore);
        Assert.Equal("Groupe 0 sélectionné", diag.FallbackSelectionReason);
        Assert.Equal(93, diag.FinalCropWidth);
        Assert.Equal(57, diag.FinalCropHeight);
        var rejected = Assert.Single(diag.FallbackRejectedCandidates);
        Assert.Equal(1, rejected.Index);
        Assert.Contains("bloc trop dense", rejected.Reason);
        Assert.Equal(800, rejected.BboxWidth);
        Assert.StartsWith("data:image/png;base64,", diag.FallbackCandidatesDataUri);
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.False(await db.ExtractedSignatures.AnyAsync());
    }

    [Fact]
    public async Task DebugMapping_LegacyDto_DefaultsToRoi()
    {
        await using var db = CreateDbContext();
        await SeedChequeAsync(db);
        _api.OnDebug = () => new SignatureDebugResponseDto
        {
            Success = true,
            ExtractionPipelineVersion = "2.4",
            OriginalWidth = 800,
            OriginalHeight = 355,
            CandidateRoi = new BoundingBoxDto { X = 320, Y = 142, Width = 464, Height = 206 },
            SignatureBbox = new BoundingBoxDto { X = 180, Y = 84, Width = 81, Height = 49 },
            ExtractionQuality = 0.7875,
        };
        var svc = CreateService(db);

        var diag = await svc.GetExtractionDiagnosticAsync(1);

        Assert.True(diag.Success);
        Assert.Equal("roi", diag.LocalizationMode);
        Assert.Equal("ROI du chèque", diag.LocalizationDisplay);
        Assert.False(diag.IsGlobalFallback);
        Assert.Equal(0, diag.FallbackCandidateCount);
        Assert.Equal(-1, diag.FallbackSelectedIndex);
        Assert.True(string.IsNullOrEmpty(diag.FallbackCandidatesDataUri));
    }

    [Fact]
    public async Task DebugMapping_MapsFallbackCandidatesTable()
    {
        await using var db = CreateDbContext();
        await SeedChequeAsync(db);
        _api.OnDebug = () => new SignatureDebugResponseDto
        {
            Success = true,
            ExtractionPipelineVersion = "2.4",
            OriginalWidth = 800,
            OriginalHeight = 355,
            CandidateRoi = new BoundingBoxDto { X = 0, Y = 0, Width = 800, Height = 355 },
            SignatureBbox = new BoundingBoxDto { X = 60, Y = 36, Width = 81, Height = 49 },
            ExtractionQuality = 0.7958,
            LocalizationMode = "global_fallback",
            LocalizationLabel = "Recherche globale de secours",
            RoiFailureReason = "ROI écartée : candidat trop compact.",
            FallbackCandidateCount = 1,
            FallbackSelectedIndex = 1,
            FallbackSelectedScore = 0.00193,
            FallbackCandidates = new List<FallbackCandidateDto>
            {
                new() { Index = 0, Bbox = new BoundingBoxDto { X = 0, Y = 0, Width = 800, Height = 61 }, Width = 800, Height = 61, Aspect = 13.115, TrueDensity = 1.0, ComponentCount = 1, InkRelative = 0.171831, VerticalExtent = 0.1718, Score = 0.029526, Status = "Rejeté — bloc graphique", Selected = false },
                new() { Index = 1, Bbox = new BoundingBoxDto { X = 60, Y = 36, Width = 81, Height = 49 }, Width = 81, Height = 49, Aspect = 1.653, TrueDensity = 0.7821, ComponentCount = 1, InkRelative = 0.013975, VerticalExtent = 0.138, Score = 0.00193, Status = "Sélectionné", Selected = true },
            },
            PrimaryRoi = new BoundingBoxDto { X = 320, Y = 142, Width = 464, Height = 206 },
            RoiCandidateFound = false,
            FallbackExecuted = true,
        };
        var svc = CreateService(db);

        var diag = await svc.GetExtractionDiagnosticAsync(1);

        Assert.True(diag.Success);
        Assert.Equal(2, diag.FallbackCandidates.Count);
        var banner = diag.FallbackCandidates[0];
        Assert.Equal(0, banner.Index);
        Assert.Equal(13.115, banner.Aspect);
        Assert.Equal(1.0, banner.TrueDensity);
        Assert.Equal("Rejeté — bloc graphique", banner.Status);
        Assert.False(banner.Selected);
        var blob = diag.FallbackCandidates[1];
        Assert.True(blob.Selected);
        Assert.Equal("Sélectionné", blob.Status);
        Assert.Equal(0.00193, blob.Score);
        Assert.Equal(320, diag.PrimaryRoiX);
        Assert.Equal(142, diag.PrimaryRoiY);
        Assert.Equal(464, diag.PrimaryRoiWidth);
        Assert.Equal(206, diag.PrimaryRoiHeight);
        Assert.False(diag.RoiCandidateFound);
        Assert.True(diag.FallbackExecuted);
        var steps = diag.LocalizationSummarySteps;
        Assert.Equal(3, steps.Count);
        Assert.Contains("aucun candidat détecté", steps[0]);
        Assert.Contains("2 candidat(s) analysé(s)", steps[1]);
        Assert.Contains("candidat #1 sélectionné", steps[2]);
    }

    [Fact]
    public async Task DebugMapping_CropCompleteness_MapsJsonContract()
    {
        await using var db = CreateDbContext();
        await SeedChequeAsync(db);
        const string json = """
            {
              "success": true,
              "extraction_pipeline_version": "2.4",
              "original_width": 800,
              "original_height": 355,
              "candidate_roi": { "x": 320, "y": 142, "width": 464, "height": 206 },
              "signature_bbox": { "x": 160, "y": 84, "width": 101, "height": 49 },
              "localization_mode": "roi",
              "crop_completeness": {
                "initial_bbox": { "x": 180, "y": 84, "width": 81, "height": 49 },
                "final_bbox": { "x": 160, "y": 84, "width": 101, "height": 49 },
                "initial_component_count": 1,
                "recovered_component_count": 1,
                "recovered_left": 1,
                "recovered_right": 0,
                "recovered_other": 0,
                "iterations": 1,
                "expansion_ratio": 1.247,
                "status": "Recadrage étendu : 1 trait voisin rattaché.",
                "recovered_strokes": [
                  {
                    "direction": "gauche",
                    "bbox": { "x": 160, "y": 84, "width": 13, "height": 39 },
                    "h_gap": 7,
                    "v_gap": 0,
                    "reason": "trait voisin plausible à gauche"
                  }
                ]
              },
              "image_format": "png",
              "original_with_roi_base64": "",
              "roi_image_base64": ""
            }
            """;
        var dto = JsonSerializer.Deserialize<SignatureDebugResponseDto>(json);
        Assert.NotNull(dto);
        _api.OnDebug = () => dto;
        var svc = CreateService(db);

        var diag = await svc.GetExtractionDiagnosticAsync(1);

        var crop = Assert.IsType<VerificationCropCompletenessViewModel>(diag.CropCompleteness);
        Assert.Equal(180, crop.InitialBboxX);
        Assert.Equal(84, crop.InitialBboxY);
        Assert.Equal(81, crop.InitialBboxWidth);
        Assert.Equal(49, crop.InitialBboxHeight);
        Assert.Equal(160, crop.FinalBboxX);
        Assert.Equal(84, crop.FinalBboxY);
        Assert.Equal(101, crop.FinalBboxWidth);
        Assert.Equal(49, crop.FinalBboxHeight);
        Assert.Equal(1, crop.InitialComponentCount);
        Assert.Equal(1, crop.RecoveredComponentCount);
        Assert.Equal(1, crop.RecoveredLeft);
        Assert.Equal(0, crop.RecoveredRight);
        Assert.Equal(0, crop.RecoveredOther);
        Assert.Equal(1, crop.Iterations);
        Assert.Equal(1.247, crop.ExpansionRatio);
        Assert.Contains("Recadrage étendu", crop.Status);
        var stroke = Assert.Single(crop.RecoveredStrokes);
        Assert.Equal("gauche", stroke.Direction);
        Assert.Equal(160, stroke.BboxX);
        Assert.Equal(84, stroke.BboxY);
        Assert.Equal(13, stroke.BboxWidth);
        Assert.Equal(39, stroke.BboxHeight);
        Assert.Equal(7, stroke.HGap);
        Assert.Equal(0, stroke.VGap);
        Assert.Contains("plausible", stroke.Reason);
    }

    [Fact]
    public void LocalizationSummarySteps_RoiMode_SingleStep()
    {
        var diag = new VerificationSignatureDebugViewModel
        {
            LocalizationMode = "roi",
            RoiCandidateFound = true,
            FallbackExecuted = false,
        };

        var steps = diag.LocalizationSummarySteps;

        Assert.Single(steps);
        Assert.Contains("candidat crédible", steps[0]);
    }

    [Fact]
    public void LocalizationSummarySteps_FallbackTextRejection()
    {
        var diag = new VerificationSignatureDebugViewModel
        {
            LocalizationMode = "global_fallback",
            RoiCandidateFound = true,
            RoiFailureReason = "ROI écartée : candidat trop compact / probablement textuel.",
            FallbackExecuted = true,
            FallbackSelectedIndex = 4,
            FallbackCandidates = new List<VerificationFallbackCandidateViewModel>
            {
                new(), new(), new(), new(), new(),
            },
        };

        var steps = diag.LocalizationSummarySteps;

        Assert.Equal(4, steps.Count);
        Assert.Contains("candidat détecté", steps[0]);
        Assert.Contains("probablement textuel", steps[1]);
        Assert.Contains("5 candidat(s) analysé(s)", steps[2]);
        Assert.Contains("candidat #4 sélectionné", steps[3]);
    }

    [Fact]
    public async Task VerifyAutomatically_ExtractionFailure_NeverCallsAi()
    {
        await using var db = CreateDbContext();
        await SeedChequeAsync(db);
        _api.OnExtract = null; // extraction API unavailable => extraction fails
        var svc = CreateService(db);

        var result = await svc.VerifyAutomaticallyAsync(1, 1);

        Assert.False(result.Success);
        Assert.Equal(0, _api.AiComparisonCallCount);
        Assert.False(await db.VerificationResults.AnyAsync());
        Assert.False(await db.ExtractedSignatures.AnyAsync());
    }

    private static ChequeVerificationDbContext CreateDbContext()
        => new(new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"verify-hybrid-{Guid.NewGuid():N}").Options);

    private VerificationService CreateService(ChequeVerificationDbContext db)
        => new(db, _api, new FakeEnv { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance);

    private static async Task SeedChequeAsync(ChequeVerificationDbContext db)
    {
        var customer = new Customer { CustomerNumber = "CUST-HYB", FullName = "Hybrid", AccountNumber = "ACC" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        var chequePath = Path.Combine(Path.GetTempPath(), "verify-hybrid-cheque-" + Guid.NewGuid().ToString("N") + ".png");
        await File.WriteAllBytesAsync(chequePath, new byte[] { 1, 2, 3 });
        db.Cheques.Add(new Cheque
        {
            CustomerId = customer.CustomerId, ImportedByUserId = 1,
            ChequeNumber = "CHQ-HYB", ImagePath = chequePath, Status = 1, UploadedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();
    }

    private sealed class FakeHybridApiClient : IVerificationApiClient
    {
        public Func<SignatureExtractionResponseDto?>? OnExtract { get; set; }
        public Func<SignatureDebugResponseDto?>? OnDebug { get; set; }
        public int AiComparisonCallCount { get; private set; }

        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default)
            => Task.FromResult<HealthResponseDto?>(null);
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult<ImageAnalysisResponseDto?>(null);
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult(OnExtract?.Invoke());
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string f, string c, CancellationToken ct = default)
            => Task.FromResult(OnDebug?.Invoke());
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default)
            => Task.FromResult<SignatureComparisonResponseDto?>(null);
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default)
        {
            AiComparisonCallCount++;
            return Task.FromResult<SignatureAiComparisonResponseDto?>(null);
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
