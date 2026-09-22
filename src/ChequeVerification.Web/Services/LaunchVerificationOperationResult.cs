namespace ChequeVerification.Web.Services;

public class LaunchVerificationOperationResult
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public int? VerificationId { get; set; }
    public decimal? SimilarityScore { get; set; }
    public byte? AutomaticDecision { get; set; }
}
