using System.Security.Claims;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Verifications;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ChequeVerification.Web.Controllers;

// Class level requires authentication only. Every action carries an
// explicit Roles policy: AuthorizeAttributes stack (all must pass), so a
// restrictive class-level Roles value would override the wider read-only
// policy on History/Details/DetailsByCheque and lock out Contrôleur.
[Authorize]
public class VerificationsController : Controller
{
    private readonly IVerificationService _verificationService;
    private readonly IVerificationApiClient _verificationApiClient;
    private readonly IWebHostEnvironment _environment;

    public VerificationsController(IVerificationService verificationService, IVerificationApiClient verificationApiClient, IWebHostEnvironment environment)
    {
        _verificationService = verificationService;
        _verificationApiClient = verificationApiClient;
        _environment = environment;
    }

    [HttpGet]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> Create(int? chequeId, CancellationToken cancellationToken = default)
    {
        var availableCheques = await _verificationService.GetAvailableChequesAsync(cancellationToken);

        var model = new VerificationCreateViewModel
        {
            SelectedChequeId = chequeId,
            AvailableCheques = availableCheques.ToList(),
            AnalysisServiceAvailable = await _verificationApiClient.GetHealthAsync(cancellationToken) != null
        };

        if (chequeId.HasValue)
        {
            var preparation = await _verificationService.PrepareVerificationAsync(chequeId.Value, cancellationToken);

            if (preparation == null)
            {
                return NotFound();
            }

            preparation.ImageIsAccessible = ResolveImageIsAccessible(preparation.ImagePath);

            foreach (var signature in preparation.ReferenceSignatures)
            {
                signature.ImageIsAccessible = ResolveImageIsAccessible(signature.ImagePath);
            }

            model.Preparation = preparation;
        }

        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> TestImage(int chequeId, CancellationToken cancellationToken = default)
    {
        var availableCheques = await _verificationService.GetAvailableChequesAsync(cancellationToken);

        var model = new VerificationCreateViewModel
        {
            SelectedChequeId = chequeId,
            AvailableCheques = availableCheques.ToList(),
            AnalysisServiceAvailable = await _verificationApiClient.GetHealthAsync(cancellationToken) != null,
            ImageTest = new VerificationImageTestViewModel { ChequeId = chequeId }
        };

        var preparation = await _verificationService.PrepareVerificationAsync(chequeId, cancellationToken);

        if (preparation == null)
        {
            return NotFound();
        }

        preparation.ImageIsAccessible = ResolveImageIsAccessible(preparation.ImagePath);

        foreach (var signature in preparation.ReferenceSignatures)
        {
            signature.ImageIsAccessible = ResolveImageIsAccessible(signature.ImagePath);
        }

        model.Preparation = preparation;

        var imagePath = await _verificationService.GetChequeImagePathAsync(chequeId, cancellationToken);
        var physicalPath = ResolvePhysicalImagePath(imagePath);

        if (physicalPath == null || !System.IO.File.Exists(physicalPath))
        {
            model.ImageTest.ErrorMessage = "L'image du chèque est introuvable sur le serveur.";
            return View(nameof(Create), model);
        }

        var extension = Path.GetExtension(physicalPath).ToLowerInvariant();
        var contentType = extension switch
        {
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            _ => "application/octet-stream"
        };

        await using var stream = new FileStream(physicalPath, FileMode.Open, FileAccess.Read, FileShare.Read);

        var result = await _verificationApiClient.AnalyzeChequeImageAsync(stream, Path.GetFileName(physicalPath), contentType, cancellationToken);

        if (result == null)
        {
            model.ImageTest.ErrorMessage = "Le service d'analyse n'a pas pu traiter l'image (indisponible ou réponse invalide).";
            return View(nameof(Create), model);
        }

        model.ImageTest.Success = result.Success;
        model.ImageTest.Width = result.Width;
        model.ImageTest.Height = result.Height;
        model.ImageTest.Channels = result.Channels;
        model.ImageTest.ContentType = result.ContentType;
        model.ImageTest.Decoded = result.Processing?.Decoded ?? false;
        model.ImageTest.GrayscaleReady = result.Processing?.GrayscaleReady ?? false;

        return View(nameof(Create), model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> TestSignatureExtraction(int chequeId, CancellationToken cancellationToken = default)
    {
        var model = await BuildCreateModelAsync(chequeId, cancellationToken);
        model.SignatureTest = new VerificationSignatureTestViewModel { ChequeId = chequeId };

        var imagePath = await _verificationService.GetChequeImagePathAsync(chequeId, cancellationToken);
        var physicalPath = ResolvePhysicalImagePath(imagePath);

        if (physicalPath == null || !System.IO.File.Exists(physicalPath))
        {
            model.SignatureTest.ErrorMessage = "L'image du chèque est introuvable sur le serveur.";
            return View(nameof(Create), model);
        }

        var extension = Path.GetExtension(physicalPath).ToLowerInvariant();
        var contentType = extension switch
        {
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            _ => "application/octet-stream"
        };

        await using var stream = new FileStream(physicalPath, FileMode.Open, FileAccess.Read, FileShare.Read);

        var result = await _verificationApiClient.ExtractSignatureAsync(stream, Path.GetFileName(physicalPath), contentType, cancellationToken);

        if (result == null)
        {
            model.SignatureTest.ErrorMessage = "Le service d'analyse n'a pas pu extraire la signature (indisponible, image inexploitable ou réponse invalide).";
            return View(nameof(Create), model);
        }

        model.SignatureTest.Success = result.Success;
        model.SignatureTest.OriginalWidth = result.OriginalWidth;
        model.SignatureTest.OriginalHeight = result.OriginalHeight;
        model.SignatureTest.RoiX = result.CandidateRoi?.X;
        model.SignatureTest.RoiY = result.CandidateRoi?.Y;
        model.SignatureTest.RoiWidth = result.CandidateRoi?.Width;
        model.SignatureTest.RoiHeight = result.CandidateRoi?.Height;
        model.SignatureTest.BboxX = result.SignatureBbox?.X;
        model.SignatureTest.BboxY = result.SignatureBbox?.Y;
        model.SignatureTest.BboxWidth = result.SignatureBbox?.Width;
        model.SignatureTest.BboxHeight = result.SignatureBbox?.Height;
        model.SignatureTest.ExtractionQuality = result.ExtractionQuality;

        if (!string.IsNullOrEmpty(result.SignatureImageBase64))
        {
            model.SignatureTest.SignatureImageDataUri = $"data:image/png;base64,{result.SignatureImageBase64}";
        }

        return View(nameof(Create), model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> DebugSignatureExtraction(int chequeId, CancellationToken cancellationToken = default)
    {
        var model = await BuildCreateModelAsync(chequeId, cancellationToken);
        model.SignatureDebug = new VerificationSignatureDebugViewModel { ChequeId = chequeId };

        var imagePath = await _verificationService.GetChequeImagePathAsync(chequeId, cancellationToken);
        var physicalPath = ResolvePhysicalImagePath(imagePath);

        if (physicalPath == null || !System.IO.File.Exists(physicalPath))
        {
            model.SignatureDebug.ErrorMessage = "L'image du chèque est introuvable sur le serveur.";
            return View(nameof(Create), model);
        }

        var extension = Path.GetExtension(physicalPath).ToLowerInvariant();
        var contentType = extension switch
        {
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            _ => "application/octet-stream"
        };

        await using var stream = new FileStream(physicalPath, FileMode.Open, FileAccess.Read, FileShare.Read);

        var result = await _verificationApiClient.DebugSignatureExtractionAsync(stream, Path.GetFileName(physicalPath), contentType, cancellationToken);

        if (result == null)
        {
            model.SignatureDebug.ErrorMessage = "Le service d'analyse n'a pas pu produire le diagnostic (indisponible, hors environnement de développement ou réponse invalide).";
            return View(nameof(Create), model);
        }

        model.SignatureDebug.Success = result.Success;
        model.SignatureDebug.PipelineVersion = result.ExtractionPipelineVersion;
        model.SignatureDebug.OriginalWidth = result.OriginalWidth;
        model.SignatureDebug.OriginalHeight = result.OriginalHeight;
        model.SignatureDebug.RoiX = result.CandidateRoi?.X;
        model.SignatureDebug.RoiY = result.CandidateRoi?.Y;
        model.SignatureDebug.RoiWidth = result.CandidateRoi?.Width;
        model.SignatureDebug.RoiHeight = result.CandidateRoi?.Height;
        model.SignatureDebug.MicrBandX = result.MicrBand?.X;
        model.SignatureDebug.MicrBandY = result.MicrBand?.Y;
        model.SignatureDebug.MicrBandWidth = result.MicrBand?.Width;
        model.SignatureDebug.MicrBandHeight = result.MicrBand?.Height;
        model.SignatureDebug.BboxX = result.SignatureBbox?.X;
        model.SignatureDebug.BboxY = result.SignatureBbox?.Y;
        model.SignatureDebug.BboxWidth = result.SignatureBbox?.Width;
        model.SignatureDebug.BboxHeight = result.SignatureBbox?.Height;
        model.SignatureDebug.TotalComponentCount = result.TotalComponentCount;
        model.SignatureDebug.RetainedComponentCount = result.RetainedComponentCount;
        model.SignatureDebug.RejectedComponentCount = result.RejectedComponentCount;
        model.SignatureDebug.MicrRejectedCount = result.MicrRejectedCount;
        model.SignatureDebug.GroupCount = result.GroupCount;
        model.SignatureDebug.SelectedGroupIndex = result.SelectedGroupIndex;
        model.SignatureDebug.SelectionReason = result.SelectionReason;
        model.SignatureDebug.RejectedComponentReasons = result.RejectedComponentReasons;
        model.SignatureDebug.Groups = result.Groups
            .Select(g => new VerificationDebugGroupViewModel
            {
                Index = g.Index,
                ComponentCount = g.ComponentCount,
                BboxX = g.Bbox?.X,
                BboxY = g.Bbox?.Y,
                BboxWidth = g.Bbox?.Width,
                BboxHeight = g.Bbox?.Height,
                InkArea = g.InkArea,
                Score = g.Score,
                Selected = g.Selected
            })
            .ToList();
        model.SignatureDebug.ExtractionQuality = result.ExtractionQuality;
        model.SignatureDebug.Message = result.Message;
        model.SignatureDebug.DominantComponentX = result.DominantComponentBbox?.X;
        model.SignatureDebug.DominantComponentY = result.DominantComponentBbox?.Y;
        model.SignatureDebug.DominantComponentWidth = result.DominantComponentBbox?.Width;
        model.SignatureDebug.DominantComponentHeight = result.DominantComponentBbox?.Height;
        model.SignatureDebug.DominantComponentInk = result.DominantComponentInk;
        model.SignatureDebug.DominantComponentScore = result.DominantComponentScore;
        model.SignatureDebug.RefinedComponentCount = result.RefinedComponentCount;
        model.SignatureDebug.CoreComponentIndices = result.CoreComponentIndices;
        model.SignatureDebug.DiscardedFromSelectedGroupIndices = result.DiscardedFromSelectedGroupIndices;
        model.SignatureDebug.RefinedBboxX = result.RefinedSignatureBbox?.X;
        model.SignatureDebug.RefinedBboxY = result.RefinedSignatureBbox?.Y;
        model.SignatureDebug.RefinedBboxWidth = result.RefinedSignatureBbox?.Width;
        model.SignatureDebug.RefinedBboxHeight = result.RefinedSignatureBbox?.Height;
        model.SignatureDebug.RefinementReason = result.RefinementReason;
        model.SignatureDebug.DirectionalAcceptedIndices = result.DirectionalAcceptedIndices;
        model.SignatureDebug.DirectionalAcceptDetails = result.DirectionalAcceptDetails
            .Select(d => new VerificationDirectionalAcceptDetailViewModel
            {
                ComponentIndex = d.ComponentIndex,
                AnchorIndex = d.AnchorIndex,
                HGap = d.HGap,
                VGap = d.VGap,
                InkDistance = d.InkDistance,
                HGapLimit = d.HGapLimit,
                InkDistanceLimit = d.InkDistanceLimit,
                Height = d.Height
            })
            .ToList();
        model.SignatureDebug.CompletenessScore = result.CompletenessScore;
        model.SignatureDebug.CompletenessRefinedInk = result.CompletenessRefinedInk;
        model.SignatureDebug.CompletenessReferenceInk = result.CompletenessReferenceInk;
        model.SignatureDebug.QualityBase = result.QualityBase;
        model.SignatureDebug.QualityCompletenessFactor = result.QualityCompletenessFactor;

        if (!string.IsNullOrEmpty(result.OriginalWithRoiBase64))
        {
            model.SignatureDebug.OriginalWithRoiDataUri = $"data:image/png;base64,{result.OriginalWithRoiBase64}";
        }

        if (!string.IsNullOrEmpty(result.RoiImageBase64))
        {
            model.SignatureDebug.RoiImageDataUri = $"data:image/png;base64,{result.RoiImageBase64}";
        }

        if (!string.IsNullOrEmpty(result.MaskImageBase64))
        {
            model.SignatureDebug.MaskImageDataUri = $"data:image/png;base64,{result.MaskImageBase64}";
        }

        if (!string.IsNullOrEmpty(result.ComponentsAllBase64))
        {
            model.SignatureDebug.ComponentsAllDataUri = $"data:image/png;base64,{result.ComponentsAllBase64}";
        }

        if (!string.IsNullOrEmpty(result.ComponentsRejectedBase64))
        {
            model.SignatureDebug.ComponentsRejectedDataUri = $"data:image/png;base64,{result.ComponentsRejectedBase64}";
        }

        if (!string.IsNullOrEmpty(result.ComponentsImageBase64))
        {
            model.SignatureDebug.ComponentsImageDataUri = $"data:image/png;base64,{result.ComponentsImageBase64}";
        }

        if (!string.IsNullOrEmpty(result.GroupImageBase64))
        {
            model.SignatureDebug.GroupImageDataUri = $"data:image/png;base64,{result.GroupImageBase64}";
        }

        if (!string.IsNullOrEmpty(result.GroupsImageBase64))
        {
            model.SignatureDebug.GroupsImageDataUri = $"data:image/png;base64,{result.GroupsImageBase64}";
        }

        if (!string.IsNullOrEmpty(result.RefinedGroupImageBase64))
        {
            model.SignatureDebug.RefinedGroupImageDataUri = $"data:image/png;base64,{result.RefinedGroupImageBase64}";
        }

        if (!string.IsNullOrEmpty(result.SignatureImageBase64))
        {
            model.SignatureDebug.SignatureImageDataUri = $"data:image/png;base64,{result.SignatureImageBase64}";
        }

        return View(nameof(Create), model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> ExtractSignature(int chequeId, CancellationToken cancellationToken = default)
    {
        var model = await BuildCreateModelAsync(chequeId, cancellationToken);

        if (model.Preparation != null)
        {
            model.Preparation.ExtractedSignatureImageIsAccessible =
                ResolveImageIsAccessible(model.Preparation.ExtractedSignatureImagePath ?? string.Empty);
        }

        model.SignatureExtraction = new VerificationSignatureExtractionViewModel { ChequeId = chequeId };

        var result = await _verificationService.ExtractAndPersistSignatureAsync(
            chequeId,
            GetCurrentUserId(),
            cancellationToken);

        model.SignatureExtraction.Success = result.Success;
        model.SignatureExtraction.Message = result.Message;
        model.SignatureExtraction.ExtractedSignatureId = result.ExtractedSignatureId;
        model.SignatureExtraction.ImagePath = result.ImagePath;
        model.SignatureExtraction.ExtractionQuality = result.ExtractionQuality;
        model.SignatureExtraction.ImageIsAccessible = ResolveImageIsAccessible(result.ImagePath);

        if (result.Success)
        {
            // The preparation above was built before the extraction was persisted.
            // Reload it so the view renders the actual persisted evidence.
            await RefreshPreparationAsync(model, chequeId, cancellationToken);
        }

        return View(nameof(Create), model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> ReExtractSignature(int chequeId, CancellationToken cancellationToken = default)
    {
        var model = await BuildCreateModelAsync(chequeId, cancellationToken);

        if (model.Preparation != null)
        {
            model.Preparation.ExtractedSignatureImageIsAccessible =
                ResolveImageIsAccessible(model.Preparation.ExtractedSignatureImagePath ?? string.Empty);
        }

        model.SignatureExtraction = new VerificationSignatureExtractionViewModel { ChequeId = chequeId };

        // Only the chequeId is accepted from the browser. Every other value
        // (ExtractedSignatureId, ImagePath, file hash, user) is resolved
        // server-side from the DB and session.
        var result = await _verificationService.ReExtractAndPersistSignatureAsync(
            chequeId,
            GetCurrentUserId(),
            cancellationToken);

        model.SignatureExtraction.Success = result.Success;
        model.SignatureExtraction.Message = result.Message;
        model.SignatureExtraction.ExtractedSignatureId = result.ExtractedSignatureId;
        model.SignatureExtraction.ImagePath = result.ImagePath;
        model.SignatureExtraction.ExtractionQuality = result.ExtractionQuality;
        model.SignatureExtraction.ImageIsAccessible = ResolveImageIsAccessible(result.ImagePath);

        if (result.Success)
        {
            // The preparation above was built before the re-extraction was persisted.
            // Reload it so the view renders the actual persisted evidence.
            await RefreshPreparationAsync(model, chequeId, cancellationToken);
        }

        return View(nameof(Create), model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> TestSignatureComparison(int chequeId, CancellationToken cancellationToken = default)
    {
        var model = await BuildCreateModelAsync(chequeId, cancellationToken);

        model.SignatureComparison = new VerificationSignatureComparisonViewModel { ChequeId = chequeId };

        var result = await _verificationService.CompareSignaturesWithReferencesAsync(chequeId, cancellationToken);

        model.SignatureComparison.Success = result.Success;
        model.SignatureComparison.Message = result.Message;
        model.SignatureComparison.Comparisons = result.Comparisons;
        model.SignatureComparison.BestReferenceSignatureId = result.BestReferenceSignatureId;
        model.SignatureComparison.BestSimilarityScore = result.BestSimilarityScore;
        model.SignatureComparison.Method = result.Method;
        model.SignatureComparison.Version = result.Version;

        return View(nameof(Create), model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> TestSignatureComparisonAi(int chequeId, CancellationToken cancellationToken = default)
    {
        var model = await BuildCreateModelAsync(chequeId, cancellationToken);

        model.SignatureAiComparison = new VerificationSignatureAiComparisonViewModel { ChequeId = chequeId };

        var result = await _verificationService.CompareAiSignaturesWithReferencesAsync(chequeId, cancellationToken);

        model.SignatureAiComparison.Success = result.Success;
        model.SignatureAiComparison.Message = result.Message;
        model.SignatureAiComparison.Comparisons = result.Comparisons;
        model.SignatureAiComparison.MeanRawScore = result.MeanRawScore;
        model.SignatureAiComparison.ActiveReferenceCount = result.ActiveReferenceCount;
        model.SignatureAiComparison.ComparedReferenceCount = result.ComparedReferenceCount;
        model.SignatureAiComparison.UnavailableReferenceCount = result.UnavailableReferenceCount;
        model.SignatureAiComparison.IsAggregationAvailable = result.IsAggregationAvailable;

        return View(nameof(Create), model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> TestChequeOcr(int chequeId, CancellationToken cancellationToken = default)
    {
        var model = await BuildCreateModelAsync(chequeId, cancellationToken);

        model.ChequeOcr = new VerificationChequeOcrViewModel { ChequeId = chequeId };

        var result = await _verificationService.OcrChequeAsync(chequeId, cancellationToken);

        model.ChequeOcr.Success = result.Success;
        model.ChequeOcr.Message = result.Message;
        model.ChequeOcr.FullText = result.FullText;
        model.ChequeOcr.Lines = result.Lines;
        model.ChequeOcr.Fields = result.Fields;
        model.ChequeOcr.ProcessingMs = result.ProcessingMs;
        model.ChequeOcr.Lang = result.Lang;
        model.ChequeOcr.Device = result.Device;

        return View(nameof(Create), model);
    }

    /// <summary>
    /// Automatic User flow: one click runs extraction (V2.4) + V5-A K=5
    /// decision via the existing service orchestration, then redirects to
    /// the complete Bilan. Only Status=1 starts (plus in-progress finish).
    /// </summary>
    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> VerifyAutomatically(int chequeId, CancellationToken cancellationToken = default)
    {
        var userId = GetCurrentUserId();
        if (userId == null)
        {
            return Forbid();
        }

        var result = await _verificationService.VerifyAutomaticallyAsync(chequeId, userId.Value, cancellationToken);

        if (result.Success && result.VerificationId.HasValue)
        {
            TempData["Success"] = result.Message;
            return RedirectToAction(nameof(Details), new { id = result.VerificationId.Value });
        }

        TempData["Error"] = result.Message;
        return RedirectToAction(nameof(Create), new { chequeId });
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> LaunchVerification(int chequeId, CancellationToken cancellationToken = default)
    {
        var userId = GetCurrentUserId();
        if (userId == null)
        {
            return Forbid();
        }

        var result = await _verificationService.LaunchVerificationAsync(chequeId, userId.Value, cancellationToken);

        if (result.Success && result.VerificationId.HasValue)
        {
            TempData["Success"] = result.Message;
            return RedirectToAction(nameof(Details), new { id = result.VerificationId.Value });
        }

        TempData["Error"] = result.Message;
        return RedirectToAction(nameof(Create), new { chequeId });
    }

    [HttpGet]
    [Authorize(Roles = "Utilisateur,Administrateur,Contrôleur")]
    public async Task<IActionResult> History(CancellationToken cancellationToken = default)
    {
        // Read-only: service performs a single AsNoTracking query over
        // persisted VerificationResult rows. No AI/OCR/OpenCV call.
        var items = await _verificationService.GetVerificationHistoryAsync(cancellationToken);
        var model = new VerificationHistoryViewModel { Items = items.ToList() };
        return View(model);
    }

    [HttpGet]
    [Authorize(Roles = "Utilisateur,Administrateur,Contrôleur")]
    public async Task<IActionResult> Details(int id, CancellationToken cancellationToken = default)
    {
        var model = await _verificationService.GetVerificationDetailsAsync(id, cancellationToken);
        if (model == null)
        {
            return NotFound();
        }

        // Bilan section 1: live diagnostic OCR snapshot. Best-effort only;
        // never persisted, never affects the persisted signature decision.
        await AttachDiagnosticOcrAsync(model, cancellationToken);
        // Bilan section 2 details: live V2.4 diagnostic. Best-effort only;
        // technical/diagnostic display, never historical evidence.
        await AttachExtractionDiagnosticAsync(model, cancellationToken);
        // Resolve image accessibility for view
        // (already done in service via WebRootPath, but re-check in case)
        return View(model);
    }

    [HttpGet]
    [Authorize(Roles = "Utilisateur,Administrateur,Contrôleur")]
    public async Task<IActionResult> DetailsByCheque(int chequeId, CancellationToken cancellationToken = default)
    {
        var model = await _verificationService.GetVerificationDetailsByChequeAsync(chequeId, cancellationToken);
        if (model == null)
        {
            return NotFound();
        }
        await AttachDiagnosticOcrAsync(model, cancellationToken);
        await AttachExtractionDiagnosticAsync(model, cancellationToken);
        return View(nameof(Details), model);
    }

    /// <summary>
    /// Best-effort diagnostic OCR for Bilan display only. Any failure yields
    /// a null Ocr (view renders "OCR indisponible"); the persisted decision
    /// is never touched.
    /// </summary>
    private async Task AttachDiagnosticOcrAsync(
        VerificationDetailsViewModel model,
        CancellationToken cancellationToken)
    {
        try
        {
            var ocr = await _verificationService.OcrChequeAsync(model.ChequeId, cancellationToken);
            model.Ocr = new VerificationChequeOcrViewModel
            {
                ChequeId = model.ChequeId,
                Success = ocr.Success,
                Message = ocr.Message,
                FullText = ocr.FullText,
                Lines = ocr.Lines,
                Fields = ocr.Fields,
                ProcessingMs = ocr.ProcessingMs,
                Lang = ocr.Lang,
                Device = ocr.Device
            };
        }
        catch
        {
            model.Ocr = null;
        }
    }

    /// <summary>
    /// Best-effort live V2.4 diagnostic for the Bilan extraction details.
    /// Any failure yields a null diagnostic (view omits the disclosure);
    /// the persisted extraction evidence and decision are never touched.
    /// </summary>
    private async Task AttachExtractionDiagnosticAsync(
        VerificationDetailsViewModel model,
        CancellationToken cancellationToken)
    {
        try
        {
            var diagnostic = await _verificationService.GetExtractionDiagnosticAsync(model.ChequeId, cancellationToken);
            model.ExtractionDiagnostic = diagnostic.Success ? diagnostic : null;
        }
        catch
        {
            model.ExtractionDiagnostic = null;
        }
    }

    /// <summary>
    /// Read-only refresh of the preparation after a successful extraction POST.
    /// The model built at action start predates the persisted row; reloading it
    /// lets the view render actual persisted evidence instead of placeholders.
    /// </summary>
    private async Task RefreshPreparationAsync(
        VerificationCreateViewModel model,
        int chequeId,
        CancellationToken cancellationToken)
    {
        var preparation = await _verificationService.PrepareVerificationAsync(chequeId, cancellationToken);
        if (preparation == null)
        {
            return;
        }

        preparation.ImageIsAccessible = ResolveImageIsAccessible(preparation.ImagePath);
        preparation.ExtractedSignatureImageIsAccessible =
            ResolveImageIsAccessible(preparation.ExtractedSignatureImagePath ?? string.Empty);
        foreach (var signature in preparation.ReferenceSignatures)
        {
            signature.ImageIsAccessible = ResolveImageIsAccessible(signature.ImagePath);
        }

        model.Preparation = preparation;
    }

    private async Task<VerificationCreateViewModel> BuildCreateModelAsync(int chequeId, CancellationToken cancellationToken)
    {
        var availableCheques = await _verificationService.GetAvailableChequesAsync(cancellationToken);

        var model = new VerificationCreateViewModel
        {
            SelectedChequeId = chequeId,
            AvailableCheques = availableCheques.ToList(),
            AnalysisServiceAvailable = await _verificationApiClient.GetHealthAsync(cancellationToken) != null
        };

        var preparation = await _verificationService.PrepareVerificationAsync(chequeId, cancellationToken);

        if (preparation != null)
        {
            preparation.ImageIsAccessible = ResolveImageIsAccessible(preparation.ImagePath);

            foreach (var signature in preparation.ReferenceSignatures)
            {
                signature.ImageIsAccessible = ResolveImageIsAccessible(signature.ImagePath);
            }

            model.Preparation = preparation;
        }

        return model;
    }

    private string? ResolvePhysicalImagePath(string? imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            return null;
        }

        if (imagePath.StartsWith("http://") || imagePath.StartsWith("https://"))
        {
            return null;
        }

        try
        {
            return imagePath.StartsWith("/") || imagePath.StartsWith("\\")
                ? Path.Combine(_environment.WebRootPath, imagePath.TrimStart('/', '\\').Replace('/', Path.DirectorySeparatorChar))
                : imagePath;
        }
        catch
        {
            return null;
        }
    }

    private int? GetCurrentUserId()
    {
        var value = User.FindFirstValue(ClaimTypes.NameIdentifier);
        return int.TryParse(value, out var id) ? id : null;
    }

    private bool ResolveImageIsAccessible(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            return false;
        }

        if (imagePath.StartsWith("http://") || imagePath.StartsWith("https://"))
        {
            return false;
        }

        try
        {
            var localPath = imagePath.StartsWith("/") || imagePath.StartsWith("\\")
                ? Path.Combine(_environment.WebRootPath, imagePath.TrimStart('/', '\\').Replace('/', Path.DirectorySeparatorChar))
                : imagePath;

            return System.IO.File.Exists(localPath);
        }
        catch
        {
            return false;
        }
    }
}