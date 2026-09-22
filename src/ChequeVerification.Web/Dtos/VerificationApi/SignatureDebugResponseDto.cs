using System.Text.Json.Serialization;

namespace ChequeVerification.Web.Dtos.VerificationApi;

public class DebugGroupInfoDto
{
    [JsonPropertyName("index")]
    public int Index { get; set; }
    [JsonPropertyName("component_count")]
    public int ComponentCount { get; set; }
    [JsonPropertyName("bbox")]
    public BoundingBoxDto? Bbox { get; set; }
    [JsonPropertyName("ink_area")]
    public int InkArea { get; set; }
    [JsonPropertyName("score")]
    public double Score { get; set; }
    [JsonPropertyName("selected")]
    public bool Selected { get; set; }
}

public class FallbackRejectedCandidateDto
{
    [JsonPropertyName("index")]
    public int Index { get; set; }
    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;
    [JsonPropertyName("bbox")]
    public BoundingBoxDto? Bbox { get; set; }
}

public class FallbackCandidateDto
{
    [JsonPropertyName("index")]
    public int Index { get; set; }
    [JsonPropertyName("bbox")]
    public BoundingBoxDto? Bbox { get; set; }
    [JsonPropertyName("width")]
    public int Width { get; set; }
    [JsonPropertyName("height")]
    public int Height { get; set; }
    [JsonPropertyName("aspect")]
    public double Aspect { get; set; }
    [JsonPropertyName("true_density")]
    public double TrueDensity { get; set; }
    [JsonPropertyName("component_count")]
    public int ComponentCount { get; set; }
    [JsonPropertyName("ink_relative")]
    public double InkRelative { get; set; }
    [JsonPropertyName("vertical_extent")]
    public double VerticalExtent { get; set; }
    [JsonPropertyName("score")]
    public double Score { get; set; }
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;
    [JsonPropertyName("selected")]
    public bool Selected { get; set; }
}

public class RecoveredStrokeDto
{
    [JsonPropertyName("direction")]
    public string Direction { get; set; } = string.Empty;
    [JsonPropertyName("bbox")]
    public BoundingBoxDto? Bbox { get; set; }
    [JsonPropertyName("h_gap")]
    public int HGap { get; set; }
    [JsonPropertyName("v_gap")]
    public int VGap { get; set; }
    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;
}

public class CropCompletenessDto
{
    [JsonPropertyName("initial_bbox")]
    public BoundingBoxDto? InitialBbox { get; set; }
    [JsonPropertyName("final_bbox")]
    public BoundingBoxDto? FinalBbox { get; set; }
    [JsonPropertyName("initial_component_count")]
    public int InitialComponentCount { get; set; }
    [JsonPropertyName("recovered_component_count")]
    public int RecoveredComponentCount { get; set; }
    [JsonPropertyName("recovered_left")]
    public int RecoveredLeft { get; set; }
    [JsonPropertyName("recovered_right")]
    public int RecoveredRight { get; set; }
    [JsonPropertyName("recovered_other")]
    public int RecoveredOther { get; set; }
    [JsonPropertyName("iterations")]
    public int Iterations { get; set; }
    [JsonPropertyName("expansion_ratio")]
    public double ExpansionRatio { get; set; } = 1.0;
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;
    [JsonPropertyName("recovered_strokes")]
    public List<RecoveredStrokeDto> RecoveredStrokes { get; set; } = new();
}

public class DirectionalAcceptDetailDto
{
    [JsonPropertyName("component_index")]
    public int ComponentIndex { get; set; }
    [JsonPropertyName("anchor_index")]
    public int AnchorIndex { get; set; }
    [JsonPropertyName("h_gap")]
    public int HGap { get; set; }
    [JsonPropertyName("v_gap")]
    public int VGap { get; set; }
    [JsonPropertyName("ink_distance")]
    public int InkDistance { get; set; }
    [JsonPropertyName("h_gap_limit")]
    public int HGapLimit { get; set; }
    [JsonPropertyName("ink_distance_limit")]
    public int InkDistanceLimit { get; set; }
    [JsonPropertyName("height")]
    public int Height { get; set; }
}

public class SignatureDebugResponseDto
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }
    // Development-only marker: which extraction pipeline version produced the response.
    [JsonPropertyName("extraction_pipeline_version")]
    public string ExtractionPipelineVersion { get; set; } = string.Empty;
    [JsonPropertyName("original_width")]
    public int OriginalWidth { get; set; }
    [JsonPropertyName("original_height")]
    public int OriginalHeight { get; set; }
    [JsonPropertyName("candidate_roi")]
    public BoundingBoxDto? CandidateRoi { get; set; }
    // Region actually analyzed (the full ROI in V2.1/V2.2).
    [JsonPropertyName("analysis_zone")]
    public BoundingBoxDto? AnalysisZone { get; set; }
    // MICR risk zone (ROI-local coordinates). Flagged, NOT physically removed.
    [JsonPropertyName("micr_band")]
    public BoundingBoxDto? MicrBand { get; set; }
    [JsonPropertyName("signature_bbox")]
    public BoundingBoxDto? SignatureBbox { get; set; }
    // Every connected component found in the ROI.
    [JsonPropertyName("total_component_count")]
    public int TotalComponentCount { get; set; }
    // Components kept after geometric filtering.
    [JsonPropertyName("retained_component_count")]
    public int RetainedComponentCount { get; set; }
    // Components dropped (all reasons).
    [JsonPropertyName("rejected_component_count")]
    public int RejectedComponentCount { get; set; }
    // Components dropped specifically as MICR / printed text.
    [JsonPropertyName("micr_rejected_count")]
    public int MicrRejectedCount { get; set; }
    // Number of spatial groups built from the kept components.
    [JsonPropertyName("group_count")]
    public int GroupCount { get; set; }
    // Per-group diagnostic info (bbox, ink, signature-likeness score).
    [JsonPropertyName("groups")]
    public List<DebugGroupInfoDto> Groups { get; set; } = new();
    // Index of the group selected as the signature candidate (-1 if none).
    [JsonPropertyName("selected_group_index")]
    public int SelectedGroupIndex { get; set; }
    // Human-readable reason for the group selection.
    [JsonPropertyName("selection_reason")]
    public string SelectionReason { get; set; } = string.Empty;
    // Count of rejected components per reason (French labels).
    [JsonPropertyName("rejected_component_reasons")]
    public Dictionary<string, int> RejectedComponentReasons { get; set; } = new();
    // Kept for backward compatibility with V2 (== retained_component_count).
    [JsonPropertyName("component_count")]
    public int ComponentCount { get; set; }
    // --- V2.3 dominant-core refinement ---
    // Bbox of the dominant signature core (ROI-local).
    [JsonPropertyName("dominant_component_bbox")]
    public BoundingBoxDto? DominantComponentBbox { get; set; }
    // True ink pixels of the dominant core.
    [JsonPropertyName("dominant_component_ink")]
    public int DominantComponentInk { get; set; }
    // Dominant-core score (ink x vertical extent x density, relative to ROI).
    [JsonPropertyName("dominant_component_score")]
    public double DominantComponentScore { get; set; }
    // Number of components kept by the V2.3 core-connected refinement.
    [JsonPropertyName("refined_component_count")]
    public int RefinedComponentCount { get; set; }
    // Indices (into the selected V2.2 group) kept by the refinement.
    [JsonPropertyName("core_component_indices")]
    public List<int> CoreComponentIndices { get; set; } = new();
    // Indices (into the selected V2.2 group) discarded by the refinement.
    [JsonPropertyName("discarded_from_selected_group_indices")]
    public List<int> DiscardedFromSelectedGroupIndices { get; set; } = new();
    // Bbox of the refined signature structure (== signature_bbox in V2.3).
    [JsonPropertyName("refined_signature_bbox")]
    public BoundingBoxDto? RefinedSignatureBbox { get; set; }
    // Human-readable reason for the V2.3 refinement.
    [JsonPropertyName("refinement_reason")]
    public string RefinementReason { get; set; } = string.Empty;
    // --- V2.4 directional continuation + completeness quality ---
    // Indices (into the selected group) accepted by the V2.4 directional pass.
    [JsonPropertyName("directional_accepted_indices")]
    public List<int> DirectionalAcceptedIndices { get; set; } = new();
    // Per-acceptance diagnostic details (anchor, gaps, ink distance, limits).
    [JsonPropertyName("directional_accept_details")]
    public List<DirectionalAcceptDetailDto> DirectionalAcceptDetails { get; set; } = new();
    // Completeness = refined ink / signature-like reference ink in [0, 1].
    [JsonPropertyName("completeness_score")]
    public double CompletenessScore { get; set; }
    // True ink pixels of the refined structure.
    [JsonPropertyName("completeness_refined_ink")]
    public int CompletenessRefinedInk { get; set; }
    // Refined ink + signature-like ink strictly right of the refined bbox.
    [JsonPropertyName("completeness_reference_ink")]
    public int CompletenessReferenceInk { get; set; }
    // V2.3 base quality before the completeness penalty.
    [JsonPropertyName("quality_base")]
    public double QualityBase { get; set; }
    // Multiplicative completeness factor actually applied to the base quality.
    [JsonPropertyName("quality_completeness_factor")]
    public double QualityCompletenessFactor { get; set; }
    [JsonPropertyName("extraction_quality")]
    public double ExtractionQuality { get; set; }
    [JsonPropertyName("message")]
    public string? Message { get; set; }
    // Hybrid localization: "roi" (expected cheque ROI) or "global_fallback"
    // (full-document fallback). Defaults keep backward compatibility.
    [JsonPropertyName("localization_mode")]
    public string LocalizationMode { get; set; } = "roi";
    [JsonPropertyName("localization_label")]
    public string LocalizationLabel { get; set; } = "ROI principale";
    [JsonPropertyName("roi_failure_reason")]
    public string RoiFailureReason { get; set; } = string.Empty;
    [JsonPropertyName("fallback_candidate_count")]
    public int FallbackCandidateCount { get; set; }
    [JsonPropertyName("fallback_selected_index")]
    public int FallbackSelectedIndex { get; set; } = -1;
    [JsonPropertyName("fallback_selected_score")]
    public double FallbackSelectedScore { get; set; }
    [JsonPropertyName("fallback_selection_reason")]
    public string FallbackSelectionReason { get; set; } = string.Empty;
    [JsonPropertyName("final_crop_width")]
    public int FinalCropWidth { get; set; }
    [JsonPropertyName("final_crop_height")]
    public int FinalCropHeight { get; set; }
    // Explainability table: one row per pre-filter fallback group, in index
    // order, with the ACTUAL ranking scores (empty when fallback never ran).
    [JsonPropertyName("fallback_candidates")]
    public List<FallbackCandidateDto> FallbackCandidates { get; set; } = new();
    // The attempted primary ROI (always set, even when the fallback won and
    // candidate_roi describes the full-document scope instead).
    [JsonPropertyName("primary_roi")]
    public BoundingBoxDto? PrimaryRoi { get; set; }
    // True when the ROI stage produced any candidate bbox (even rejected).
    [JsonPropertyName("roi_candidate_found")]
    public bool RoiCandidateFound { get; set; }
    // True as soon as the full-document fallback executed.
    [JsonPropertyName("fallback_executed")]
    public bool FallbackExecuted { get; set; }
    // Crop-completeness evidence for the final candidate (null when no
    // refined bbox exists).
    [JsonPropertyName("crop_completeness")]
    public CropCompletenessDto? CropCompleteness { get; set; }
    // Whole fallback groups rejected as obvious graphics, with truthful
    // reasons (empty when the ROI won or nothing was graphically rejected).
    [JsonPropertyName("fallback_rejected_candidates")]
    public List<FallbackRejectedCandidateDto> FallbackRejectedCandidates { get; set; } = new();
    [JsonPropertyName("image_format")]
    public string ImageFormat { get; set; } = string.Empty;
    [JsonPropertyName("original_with_roi_base64")]
    public string OriginalWithRoiBase64 { get; set; } = string.Empty;
    [JsonPropertyName("roi_image_base64")]
    public string RoiImageBase64 { get; set; } = string.Empty;
    // Binary ink mask of the full ROI (255 = ink).
    [JsonPropertyName("mask_image_base64")]
    public string? MaskImageBase64 { get; set; }
    // ROI crop with every detected component drawn (distinct colors).
    [JsonPropertyName("components_all_base64")]
    public string? ComponentsAllBase64 { get; set; }
    // ROI crop with the rejected components drawn (gray; MICR ones in orange).
    [JsonPropertyName("components_rejected_base64")]
    public string? ComponentsRejectedBase64 { get; set; }
    // ROI crop with the retained components drawn (kept name for compat).
    [JsonPropertyName("components_image_base64")]
    public string? ComponentsImageBase64 { get; set; }
    // ROI crop with the final signature group drawn (distinct colors) + bbox.
    [JsonPropertyName("group_image_base64")]
    public string? GroupImageBase64 { get; set; }
    // ROI crop with all generated groups (each in a distinct color) + bbox.
    [JsonPropertyName("groups_image_base64")]
    public string? GroupsImageBase64 { get; set; }
    // ROI crop with the V2.3 refined structure (core yellow, refined kept
    // components, discarded-from-selected-group in gray) + refined bbox red.
    [JsonPropertyName("refined_group_image_base64")]
    public string? RefinedGroupImageBase64 { get; set; }
    // Full-document fallback candidates (each group a distinct color, selected
    // region red). Only present when the fallback ran.
    [JsonPropertyName("fallback_candidates_image_base64")]
    public string? FallbackCandidatesImageBase64 { get; set; }
    [JsonPropertyName("signature_image_base64")]
    public string? SignatureImageBase64 { get; set; }
}