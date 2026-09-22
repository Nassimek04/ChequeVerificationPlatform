namespace ChequeVerification.Web.Services;

public class SignatureExtractionOperationResult
{
    public bool Success { get; set; }

    public string Message { get; set; } = string.Empty;

    public int ExtractedSignatureId { get; set; }

    public string ImagePath { get; set; } = string.Empty;

    public decimal? ExtractionQuality { get; set; }
}