namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationPreparationViewModel
{
    public int ChequeId { get; set; }
    public string ChequeNumber { get; set; } = string.Empty;
    public int CustomerId { get; set; }
    public string CustomerNumber { get; set; } = string.Empty;
    public string CustomerFullName { get; set; } = string.Empty;
    public string AccountNumber { get; set; } = string.Empty;
    public decimal? Amount { get; set; }
    public DateOnly? IssueDate { get; set; }
    public string ImagePath { get; set; } = string.Empty;
    public byte Status { get; set; }
    public DateTime UploadedAt { get; set; }
    public bool ImageIsAccessible { get; set; }
    public bool CanStartVerification { get; set; }
    public string? BlockingReason { get; set; }
    public bool HasExistingVerification { get; set; }
    public List<VerificationReferenceSignatureViewModel> ReferenceSignatures { get; set; } = new();
    public bool HasExtractedSignature { get; set; }
    public int? ExtractedSignatureId { get; set; }
    public string? ExtractedSignatureImagePath { get; set; }
    public decimal? ExtractedSignatureQuality { get; set; }
    public bool ExtractedSignatureImageIsAccessible { get; set; }
}