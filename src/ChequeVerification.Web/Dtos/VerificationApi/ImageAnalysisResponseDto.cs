using System.Text.Json.Serialization;

namespace ChequeVerification.Web.Dtos.VerificationApi;

public class ImageAnalysisResponseDto
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }
    [JsonPropertyName("width")]
    public int Width { get; set; }
    [JsonPropertyName("height")]
    public int Height { get; set; }
    [JsonPropertyName("channels")]
    public int Channels { get; set; }
    [JsonPropertyName("content_type")]
    public string ContentType { get; set; } = string.Empty;
    [JsonPropertyName("processing")]
    public ImageAnalysisProcessingDto? Processing { get; set; }
}

public class ImageAnalysisProcessingDto
{
    [JsonPropertyName("decoded")]
    public bool Decoded { get; set; }
    [JsonPropertyName("grayscale_ready")]
    public bool GrayscaleReady { get; set; }
}