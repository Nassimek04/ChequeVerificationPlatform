namespace ChequeVerification.Web.ViewModels.Verifications;

// Read-only history item. Every field is copied verbatim from the persisted
// VerificationResult row (including the thresholds stored at verification
// time). Never reinterpreted with current policy values.
public class VerificationHistoryItemViewModel
{
    public int VerificationId { get; set; }
    public int ChequeId { get; set; }
    public string ChequeNumber { get; set; } = string.Empty;
    public int CustomerId { get; set; }
    public string CustomerNumber { get; set; } = string.Empty;
    public string CustomerFullName { get; set; } = string.Empty;
    public DateTime VerifiedAt { get; set; }
    public decimal SimilarityScore { get; set; }
    public decimal LowerThresholdUsed { get; set; }
    public decimal UpperThresholdUsed { get; set; }
    public byte AutomaticDecision { get; set; }
    public byte? FinalDecision { get; set; }
    public string? ModelName { get; set; }
    public string? ModelVersion { get; set; }
    public int? ReviewedByUserId { get; set; }
    public string? ReviewedByName { get; set; }
    public string? ReviewerComment { get; set; }
}

public class VerificationHistoryViewModel
{
    public List<VerificationHistoryItemViewModel> Items { get; set; } = new();
}
