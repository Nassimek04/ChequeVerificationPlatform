using ChequeVerification.Web.ViewModels.Dashboard;
using ChequeVerification.Web.Services;

namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationDetailsViewModel
{
    public int VerificationId { get; set; }
    public int ChequeId { get; set; }
    public string ChequeNumber { get; set; } = string.Empty;
    public int CustomerId { get; set; }
    public string CustomerNumber { get; set; } = string.Empty;
    public string CustomerFullName { get; set; } = string.Empty;
    public byte ChequeStatus { get; set; }
    public string ChequeImagePath { get; set; } = string.Empty;
    public bool ChequeImageIsAccessible { get; set; }
    public string? ExtractedSignatureImagePath { get; set; }
    public bool ExtractedSignatureImageIsAccessible { get; set; }
    public decimal? ExtractionQuality { get; set; }
    public int? ExtractedSignatureId { get; set; }
    public DateTime? ExtractedAt { get; set; }

    /// <summary>
    /// Live V2.4 diagnostic snapshot for the Bilan extraction details.
    /// Rerun on demand via the debug endpoint: technical/diagnostic only,
    /// never historical evidence and never used in the decision.
    /// </summary>
    public VerificationSignatureDebugViewModel? ExtractionDiagnostic { get; set; }

    /// <summary>
    /// Live diagnostic OCR snapshot for Bilan section 1. Never persisted and
    /// never used in the signature decision.
    /// </summary>
    public VerificationChequeOcrViewModel? Ocr { get; set; }

    public decimal SimilarityScore { get; set; }
    public decimal LowerThresholdUsed { get; set; }
    public decimal UpperThresholdUsed { get; set; }
    public byte AutomaticDecision { get; set; }
    public byte? FinalDecision { get; set; }
    public string? ModelName { get; set; }
    public string? ModelVersion { get; set; }
    public DateTime VerifiedAt { get; set; }
    public int? ReviewedByUserId { get; set; }
    public string? ReviewedByName { get; set; }
    public string? ReviewerComment { get; set; }

    public int ActiveReferenceCount { get; set; }
    public int ComparedReferenceCount { get; set; }

    public List<VerificationDetailsComparisonViewModel> Comparisons { get; set; } = new();

    public bool RequiresManualReview => AutomaticDecision == VerificationDecision.ControleManuel && FinalDecision == null;
}

public class VerificationDetailsComparisonViewModel
{
    public int ComparisonId { get; set; }
    public int ReferenceSignatureId { get; set; }
    public string ReferenceImagePath { get; set; } = string.Empty;
    public bool ReferenceImageIsAccessible { get; set; }
    public decimal SimilarityScore { get; set; }
    public bool IsBestMatch { get; set; }
    public DateTime ReferenceCreatedAt { get; set; }
}
