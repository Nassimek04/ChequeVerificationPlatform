namespace ChequeVerification.Web.ViewModels.ManualReviews;

public class ManualReviewDetailsViewModel
{
    // Verification
    public int VerificationId { get; set; }
    public int ChequeId { get; set; }
    public DateTime VerifiedAt { get; set; }
    public decimal SimilarityScore { get; set; }
    public decimal LowerThresholdUsed { get; set; }
    public decimal UpperThresholdUsed { get; set; }
    public byte AutomaticDecision { get; set; }
    public byte? FinalDecision { get; set; }
    public string? ModelName { get; set; }
    public string? ModelVersion { get; set; }

    // Cheque
    public string ChequeNumber { get; set; } = string.Empty;
    public decimal? Amount { get; set; }
    public DateOnly? IssueDate { get; set; }
    public string ChequeImagePath { get; set; } = string.Empty;
    public bool ChequeImageIsAccessible { get; set; }
    public byte ChequeStatus { get; set; }

    // Customer
    public int CustomerId { get; set; }
    public string CustomerNumber { get; set; } = string.Empty;
    public string CustomerFullName { get; set; } = string.Empty;
    public string AccountNumber { get; set; } = string.Empty;

    // Extracted signature
    public string? ExtractedSignatureImagePath { get; set; }
    public bool ExtractedSignatureIsAccessible { get; set; }
    public decimal? ExtractionConfidence { get; set; }

    // References
    public List<ManualReviewReferenceViewModel> References { get; set; } = new();

    // For decision form
    public ManualReviewDecisionViewModel Decision { get; set; } = new();
}

public class ManualReviewReferenceViewModel
{
    public int ReferenceSignatureId { get; set; }
    public string ImagePath { get; set; } = string.Empty;
    public bool ImageIsAccessible { get; set; }
    public decimal SimilarityScore { get; set; }
    public bool IsBestMatch { get; set; }
    public DateTime CreatedAt { get; set; }
}
