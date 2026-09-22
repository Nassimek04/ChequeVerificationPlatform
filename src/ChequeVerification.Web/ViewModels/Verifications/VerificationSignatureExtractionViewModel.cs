namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationSignatureExtractionViewModel
{
    public int ChequeId { get; set; }
    public bool Success { get; set; }
    public string? Message { get; set; }
    public int ExtractedSignatureId { get; set; }
    public string ImagePath { get; set; } = string.Empty;
    public decimal? ExtractionQuality { get; set; }
    public bool ImageIsAccessible { get; set; }
}