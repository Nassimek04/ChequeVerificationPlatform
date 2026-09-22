using ChequeVerification.Web.ViewModels.ManualReviews;

namespace ChequeVerification.Web.Services.Interfaces;

public interface IManualReviewService
{
    Task<IReadOnlyList<ManualReviewQueueItemViewModel>> GetPendingQueueAsync(CancellationToken cancellationToken = default);
    Task<ManualReviewDetailsViewModel?> GetReviewDetailsAsync(int verificationId, CancellationToken cancellationToken = default);
    Task<ManualReviewDecisionResult> DecideAsync(int verificationId, byte finalDecision, string reviewerComment, int reviewerUserId, CancellationToken cancellationToken = default);
}

public class ManualReviewDecisionResult
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public int? VerificationId { get; set; }
}
