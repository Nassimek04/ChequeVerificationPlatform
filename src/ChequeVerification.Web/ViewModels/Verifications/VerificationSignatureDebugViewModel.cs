namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationDebugGroupViewModel
{
    public int Index { get; set; }
    public int ComponentCount { get; set; }
    public int? BboxX { get; set; }
    public int? BboxY { get; set; }
    public int? BboxWidth { get; set; }
    public int? BboxHeight { get; set; }
    public int InkArea { get; set; }
    public double Score { get; set; }
    public bool Selected { get; set; }
}

public class VerificationDirectionalAcceptDetailViewModel
{
    public int ComponentIndex { get; set; }
    public int AnchorIndex { get; set; }
    public int HGap { get; set; }
    public int VGap { get; set; }
    public int InkDistance { get; set; }
    public int HGapLimit { get; set; }
    public int InkDistanceLimit { get; set; }
    public int Height { get; set; }
}

public class VerificationRecoveredStrokeViewModel
{
    public string Direction { get; set; } = string.Empty;
    public int? BboxX { get; set; }
    public int? BboxY { get; set; }
    public int? BboxWidth { get; set; }
    public int? BboxHeight { get; set; }
    public int HGap { get; set; }
    public int VGap { get; set; }
    public string Reason { get; set; } = string.Empty;
}

public class VerificationCropCompletenessViewModel
{
    public int? InitialBboxX { get; set; }
    public int? InitialBboxY { get; set; }
    public int? InitialBboxWidth { get; set; }
    public int? InitialBboxHeight { get; set; }
    public int? FinalBboxX { get; set; }
    public int? FinalBboxY { get; set; }
    public int? FinalBboxWidth { get; set; }
    public int? FinalBboxHeight { get; set; }
    public int InitialComponentCount { get; set; }
    public int RecoveredComponentCount { get; set; }
    public int RecoveredLeft { get; set; }
    public int RecoveredRight { get; set; }
    public int RecoveredOther { get; set; }
    public int Iterations { get; set; }
    public double ExpansionRatio { get; set; } = 1.0;
    public string Status { get; set; } = string.Empty;
    public List<VerificationRecoveredStrokeViewModel> RecoveredStrokes { get; set; } = new();
}

public class VerificationFallbackRejectedCandidateViewModel
{
    public int Index { get; set; }
    public string Reason { get; set; } = string.Empty;
    public int? BboxX { get; set; }
    public int? BboxY { get; set; }
    public int? BboxWidth { get; set; }
    public int? BboxHeight { get; set; }
}

public class VerificationFallbackCandidateViewModel
{
    public int Index { get; set; }
    public int? BboxX { get; set; }
    public int? BboxY { get; set; }
    public int? BboxWidth { get; set; }
    public int? BboxHeight { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public double Aspect { get; set; }
    public double TrueDensity { get; set; }
    public int ComponentCount { get; set; }
    public double InkRelative { get; set; }
    public double VerticalExtent { get; set; }
    public double Score { get; set; }
    public string Status { get; set; } = string.Empty;
    public bool Selected { get; set; }
}

public class VerificationSignatureDebugViewModel
{
    public int ChequeId { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    // Development-only marker: which extraction pipeline version produced the diagnostic.
    public string PipelineVersion { get; set; } = string.Empty;
    public int? OriginalWidth { get; set; }
    public int? OriginalHeight { get; set; }
    public int? RoiX { get; set; }
    public int? RoiY { get; set; }
    public int? RoiWidth { get; set; }
    public int? RoiHeight { get; set; }
    // MICR risk band (ROI-local coordinates).
    public int? MicrBandX { get; set; }
    public int? MicrBandY { get; set; }
    public int? MicrBandWidth { get; set; }
    public int? MicrBandHeight { get; set; }
    public int? BboxX { get; set; }
    public int? BboxY { get; set; }
    public int? BboxWidth { get; set; }
    public int? BboxHeight { get; set; }
    public int? TotalComponentCount { get; set; }
    public int? RetainedComponentCount { get; set; }
    public int? RejectedComponentCount { get; set; }
    public int? MicrRejectedCount { get; set; }
    public int? GroupCount { get; set; }
    public int? SelectedGroupIndex { get; set; }
    public string SelectionReason { get; set; } = string.Empty;
    public Dictionary<string, int> RejectedComponentReasons { get; set; } = new();
    public List<VerificationDebugGroupViewModel> Groups { get; set; } = new();
    public double? ExtractionQuality { get; set; }
    // --- V2.3 dominant-core refinement ---
    public int? DominantComponentX { get; set; }
    public int? DominantComponentY { get; set; }
    public int? DominantComponentWidth { get; set; }
    public int? DominantComponentHeight { get; set; }
    public int? DominantComponentInk { get; set; }
    public double? DominantComponentScore { get; set; }
    public int? RefinedComponentCount { get; set; }
    public List<int> CoreComponentIndices { get; set; } = new();
    public List<int> DiscardedFromSelectedGroupIndices { get; set; } = new();
    public int? RefinedBboxX { get; set; }
    public int? RefinedBboxY { get; set; }
    public int? RefinedBboxWidth { get; set; }
    public int? RefinedBboxHeight { get; set; }
    public string RefinementReason { get; set; } = string.Empty;
    // --- V2.4 directional continuation + completeness quality ---
    public List<int> DirectionalAcceptedIndices { get; set; } = new();
    public List<VerificationDirectionalAcceptDetailViewModel> DirectionalAcceptDetails { get; set; } = new();
    public double? CompletenessScore { get; set; }
    public int? CompletenessRefinedInk { get; set; }
    public int? CompletenessReferenceInk { get; set; }
    public double? QualityBase { get; set; }
    public double? QualityCompletenessFactor { get; set; }
    public string? Message { get; set; }
    // --- Hybrid localization (ROI first, full-document fallback) ---
    // Machine mode: "roi" or "global_fallback".
    public string LocalizationMode { get; set; } = "roi";
    // French label for the Bilan ("ROI principale" / "Recherche globale de secours").
    public string LocalizationLabel { get; set; } = "ROI principale";
    // Why the ROI candidate was rejected (empty when the ROI won).
    public string RoiFailureReason { get; set; } = string.Empty;
    public int FallbackCandidateCount { get; set; }
    public int FallbackSelectedIndex { get; set; } = -1;
    public double FallbackSelectedScore { get; set; }
    public string FallbackSelectionReason { get; set; } = string.Empty;
    public int FinalCropWidth { get; set; }
    public int FinalCropHeight { get; set; }
    // Whole fallback groups rejected as obvious graphics, with truthful
    // reasons (diagnostic only, not rendered in the business Bilan).
    public List<VerificationFallbackRejectedCandidateViewModel> FallbackRejectedCandidates { get; set; } = new();
    // Explainability table: one row per pre-filter fallback group (empty
    // when the fallback never ran). Rendered in technical diagnostics only.
    public List<VerificationFallbackCandidateViewModel> FallbackCandidates { get; set; } = new();
    // The attempted primary ROI (always set by the API).
    public int? PrimaryRoiX { get; set; }
    public int? PrimaryRoiY { get; set; }
    public int? PrimaryRoiWidth { get; set; }
    public int? PrimaryRoiHeight { get; set; }
    // True when the ROI stage produced any candidate bbox (even rejected).
    public bool RoiCandidateFound { get; set; }
    // True as soon as the full-document fallback executed.
    public bool FallbackExecuted { get; set; }
    public VerificationCropCompletenessViewModel? CropCompleteness { get; set; }
    /// <summary>Bilan display value: "ROI du chèque" or "Recherche globale automatique".</summary>
    public string LocalizationDisplay => LocalizationMode == "global_fallback"
        ? "Recherche globale automatique"
        : "ROI du chèque";
    public bool IsGlobalFallback => LocalizationMode == "global_fallback";
    /// <summary>
    /// Short human-readable motive for the ROI rejection, derived from the
    /// actual diagnostic reason (full text stays in technical details).
    /// </summary>
    public string RoiRejectionShort
    {
        get
        {
            var r = RoiFailureReason ?? string.Empty;
            if (r.Contains("textuel")) return "candidat rejeté comme probablement textuel";
            if (r.Contains("peu compatible")) return "candidat rejeté (géométrie peu compatible)";
            if (r.Contains("trait incompatible")) return "candidat rejeté (trait incompatible)";
            if (r.Contains("qualité")) return "candidat rejeté (qualité insuffisante)";
            if (string.IsNullOrWhiteSpace(r)) return "candidat écarté";
            return "candidat écarté — motif technique ci-dessous";
        }
    }
    /// <summary>
    /// Compact localization decision summary, rendered from actual
    /// diagnostic data (no hard-coded case values).
    /// </summary>
    public List<string> LocalizationSummarySteps
    {
        get
        {
            if (!IsGlobalFallback)
                return new List<string> { "ROI principale → candidat crédible → extraction réussie" };
            var steps = new List<string>
            {
                "ROI principale → " + (RoiCandidateFound ? "candidat détecté" : "aucun candidat détecté")
            };
            if (RoiCandidateFound)
                steps.Add(RoiRejectionShort);
            steps.Add($"recherche globale déclenchée → {FallbackCandidates.Count} candidat(s) analysé(s)");
            steps.Add($"candidat #{FallbackSelectedIndex} sélectionné → extraction réussie");
            return steps;
        }
    }
    public string OriginalWithRoiDataUri { get; set; } = string.Empty;
    public string RoiImageDataUri { get; set; } = string.Empty;
    public string MaskImageDataUri { get; set; } = string.Empty;
    public string ComponentsAllDataUri { get; set; } = string.Empty;
    public string ComponentsRejectedDataUri { get; set; } = string.Empty;
    public string ComponentsImageDataUri { get; set; } = string.Empty;
    public string GroupImageDataUri { get; set; } = string.Empty;
    public string GroupsImageDataUri { get; set; } = string.Empty;
    public string RefinedGroupImageDataUri { get; set; } = string.Empty;
    public string FallbackCandidatesDataUri { get; set; } = string.Empty;
    public string SignatureImageDataUri { get; set; } = string.Empty;
}
