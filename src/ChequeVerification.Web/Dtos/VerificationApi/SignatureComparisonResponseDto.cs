using System.Text.Json.Serialization;

namespace ChequeVerification.Web.Dtos.VerificationApi;

public class SignatureComparisonMetricsDto
{
    [JsonPropertyName("mask_overlap")]
    public double MaskOverlap { get; set; }
    [JsonPropertyName("normalized_correlation")]
    public double NormalizedCorrelation { get; set; }
    [JsonPropertyName("density_similarity")]
    public double DensitySimilarity { get; set; }
}

public class SignatureComparisonResponseDto
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }
    [JsonPropertyName("similarity_score")]
    public double SimilarityScore { get; set; }
    [JsonPropertyName("method")]
    public string Method { get; set; } = string.Empty;
    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;
    [JsonPropertyName("metrics")]
    public SignatureComparisonMetricsDto? Metrics { get; set; }
}
