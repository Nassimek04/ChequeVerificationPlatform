using System.Text.Json.Serialization;

namespace ChequeVerification.Web.Dtos.VerificationApi;

public class HealthResponseDto
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;
    [JsonPropertyName("service")]
    public string Service { get; set; } = string.Empty;
    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;
}