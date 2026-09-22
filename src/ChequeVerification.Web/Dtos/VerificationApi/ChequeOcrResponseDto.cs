using System.Text.Json.Serialization;

namespace ChequeVerification.Web.Dtos.VerificationApi;

public class ChequeOcrLineDto
{
    [JsonPropertyName("text")]
    public string Text { get; set; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("box")]
    public List<List<double>> Box { get; set; } = new();
}

public class ChequeOcrFieldsDto
{
    [JsonPropertyName("cheque_number")]
    public string? ChequeNumber { get; set; }

    [JsonPropertyName("date")]
    public string? Date { get; set; }

    [JsonPropertyName("amount_text")]
    public string? AmountText { get; set; }

    [JsonPropertyName("amount_numeric")]
    public string? AmountNumeric { get; set; }

    [JsonPropertyName("account_number")]
    public string? AccountNumber { get; set; }

    // CMC7 constrained parse (additive; null/false when unavailable/invalid).
    [JsonPropertyName("cmc7_raw")]
    public string? Cmc7Raw { get; set; }

    [JsonPropertyName("cmc7_cheque_number")]
    public string? Cmc7ChequeNumber { get; set; }

    [JsonPropertyName("cmc7_account_number")]
    public string? Cmc7AccountNumber { get; set; }

    [JsonPropertyName("cmc7_valid")]
    public bool Cmc7Valid { get; set; }

    [JsonPropertyName("cmc7_error")]
    public string? Cmc7Error { get; set; }

    [JsonPropertyName("cmc7_trailing_noise")]
    public bool Cmc7TrailingNoise { get; set; }

    [JsonPropertyName("printed_cheque_number")]
    public string? PrintedChequeNumber { get; set; }

    [JsonPropertyName("printed_account_number")]
    public string? PrintedAccountNumber { get; set; }

    [JsonPropertyName("cmc7_cross_check")]
    public string? Cmc7CrossCheck { get; set; }

    [JsonPropertyName("is_probable_verso")]
    public bool IsProbableVerso { get; set; }
}

public class ChequeOcrResponseDto
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("full_text")]
    public string FullText { get; set; } = string.Empty;

    [JsonPropertyName("lines")]
    public List<ChequeOcrLineDto> Lines { get; set; } = new();

    [JsonPropertyName("fields")]
    public ChequeOcrFieldsDto Fields { get; set; } = new();

    [JsonPropertyName("processing_ms")]
    public int ProcessingMs { get; set; }

    [JsonPropertyName("lang")]
    public string Lang { get; set; } = string.Empty;

    [JsonPropertyName("device")]
    public string Device { get; set; } = string.Empty;
}
