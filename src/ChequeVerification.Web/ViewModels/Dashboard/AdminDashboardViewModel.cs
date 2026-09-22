namespace ChequeVerification.Web.ViewModels.Dashboard;

public class AdminDashboardViewModel
{
    public int TotalUsers { get; set; }
    public int TotalCustomers { get; set; }
    public int TotalCheques { get; set; }
    public int TotalVerifications { get; set; }
    public List<AuditLogItemViewModel> RecentActivity { get; set; } = new();
    public VerificationActivitySeries Activity { get; set; } = new();
    public DecisionDistribution Decisions { get; set; } = new();
    public ScoreZoneDistribution ScoreZones { get; set; } = new();
    public ChequeStatusDistribution ChequePipeline { get; set; } = new();
    public List<VerificationItemViewModel> RecentVerifications { get; set; } = new();
}