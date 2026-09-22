namespace ChequeVerification.Web.ViewModels.Dashboard;

public class VerificationItemViewModel
{
    public int VerificationId { get; set; }
    public string ChequeNumber { get; set; } = string.Empty;
    public string CustomerName { get; set; } = string.Empty;
    public decimal SimilarityScore { get; set; }
    public byte AutomaticDecision { get; set; }
    public byte? FinalDecision { get; set; }
    public DateTime VerifiedAt { get; set; }
}
