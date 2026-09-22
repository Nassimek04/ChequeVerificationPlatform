namespace ChequeVerification.Web.ViewModels.ManualReviews;

public class ManualReviewQueueViewModel
{
    public IReadOnlyList<ManualReviewQueueItemViewModel> Items { get; set; } = Array.Empty<ManualReviewQueueItemViewModel>();
    public int TotalCount => Items.Count;
}
