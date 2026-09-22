namespace ChequeVerification.Web.ViewModels.Cheques;

public class ChequeDetailsViewModel
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
    public string ImportedByFullName { get; set; } = string.Empty;
    public bool HasExtractedSignature { get; set; }
    public bool HasVerificationResult { get; set; }
    public bool ImageIsAccessible { get; set; }
}