namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationChequeOptionViewModel
{
    public int ChequeId { get; set; }
    public string ChequeNumber { get; set; } = string.Empty;
    public string CustomerNumber { get; set; } = string.Empty;
    public string CustomerFullName { get; set; } = string.Empty;
    public string AccountNumber { get; set; } = string.Empty;
    public DateOnly? IssueDate { get; set; }
    public decimal? Amount { get; set; }
    public byte Status { get; set; }
}