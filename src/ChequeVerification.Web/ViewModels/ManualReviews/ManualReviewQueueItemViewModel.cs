namespace ChequeVerification.Web.ViewModels.ManualReviews;

public class ManualReviewQueueItemViewModel
{
    public int VerificationId { get; set; }
    public int ChequeId { get; set; }
    public string ChequeNumber { get; set; } = string.Empty;
    public int CustomerId { get; set; }
    public string CustomerNumber { get; set; } = string.Empty;
    public string CustomerFullName { get; set; } = string.Empty;
    public DateTime VerifiedAt { get; set; }
    public decimal SimilarityScore { get; set; }
    public string? ModelName { get; set; }
    public string? ModelVersion { get; set; }
}
