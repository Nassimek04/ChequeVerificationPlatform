namespace ChequeVerification.Web.ViewModels.Dashboard;

public class UserDashboardViewModel
{
    public int TotalCheques { get; set; }
    public int TotalVerifications { get; set; }
    public int ConformeCount { get; set; }
    public int ManualReviewCount { get; set; }
    public List<VerificationItemViewModel> LatestVerifications { get; set; } = new();
    public VerificationActivitySeries Activity { get; set; } = new();
}