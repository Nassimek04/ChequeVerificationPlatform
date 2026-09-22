using System.Text.Json.Serialization;

namespace ChequeVerification.Web.Dtos.VerificationApi;

public class SignatureAiInkBBoxDto
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

// Mirrors FastAPI AiSignatureComparisonResponse (AI V2). The score is a raw
// cosine similarity in [-1;1]: a technical measure, never a probability, a
// threshold decision or a conformity verdict. The 503 "AI unavailable"
// response uses the same shape with Success=false and SimilarityScore=null.
public class SignatureAiComparisonResponseDto
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }
    [JsonPropertyName("similarity_score")]
    public double? SimilarityScore { get; set; }
    [JsonPropertyName("method")]
    public string Method { get; set; } = string.Empty;
    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;
    [JsonPropertyName("embedding_dimension")]
    public int EmbeddingDimension { get; set; }
    [JsonPropertyName("device")]
    public string Device { get; set; } = string.Empty;
    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;
    [JsonPropertyName("extracted_ink_bbox")]
    public SignatureAiInkBBoxDto? ExtractedInkBBox { get; set; }
    [JsonPropertyName("reference_ink_bbox")]
    public SignatureAiInkBBoxDto? ReferenceInkBBox { get; set; }
}
