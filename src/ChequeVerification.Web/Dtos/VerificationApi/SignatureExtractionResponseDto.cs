using System.Text.Json.Serialization;

namespace ChequeVerification.Web.Dtos.VerificationApi;

public class BoundingBoxDto
{
    [JsonPropertyName("x")]
    public int X { get; set; }
    [JsonPropertyName("y")]
    public int Y { get; set; }
    [JsonPropertyName("width")]
    public int Width { get; set; }
    [JsonPropertyName("height")]
    public int Height { get; set; }
}

public class SignatureExtractionResponseDto
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }
    [JsonPropertyName("original_width")]
    public int OriginalWidth { get; set; }
    [JsonPropertyName("original_height")]
    public int OriginalHeight { get; set; }
    [JsonPropertyName("candidate_roi")]
    public BoundingBoxDto? CandidateRoi { get; set; }
    [JsonPropertyName("signature_bbox")]
    public BoundingBoxDto? SignatureBbox { get; set; }
    [JsonPropertyName("extraction_quality")]
    public double ExtractionQuality { get; set; }
    [JsonPropertyName("image_format")]
    public string ImageFormat { get; set; } = string.Empty;
    [JsonPropertyName("signature_image_base64")]
    public string SignatureImageBase64 { get; set; } = string.Empty;
    // Hybrid localization: "roi" (expected cheque ROI) or "global_fallback"
    // (full-document fallback). Defaults keep backward compatibility.
    [JsonPropertyName("localization_mode")]
    public string LocalizationMode { get; set; } = "roi";
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
}