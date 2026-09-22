namespace ChequeVerification.Web.ViewModels.Dashboard;

public class ControllerDashboardViewModel
{
    public int PendingReviewCount { get; set; }
    public int RecentDecisionsCount { get; set; }
    public int ValidatedCount { get; set; }
    public int RejectedCount { get; set; }
    /// <summary>
    /// Dossiers decided by a human controller (ReviewedByUserId set by the
    /// manual-review workflow). Subset of the finalized decisions; the rest
    /// were finalized automatically at verification creation.
    /// </summary>
    public int ManualReviewCompletedCount { get; set; }
    public List<VerificationItemViewModel> LatestVerificationsToReview { get; set; } = new();
    public DecisionDistribution ControlDecisions { get; set; } = new();
}