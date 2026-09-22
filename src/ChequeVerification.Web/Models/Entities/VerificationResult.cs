using System;
using System.Collections.Generic;

namespace ChequeVerification.Web.Models.Entities;

public partial class VerificationResult
{
    public int VerificationId { get; set; }

    public int ChequeId { get; set; }

    public int? ReviewedByUserId { get; set; }

    public decimal SimilarityScore { get; set; }

    public decimal LowerThresholdUsed { get; set; }

    public decimal UpperThresholdUsed { get; set; }

    public byte AutomaticDecision { get; set; }

    public byte? FinalDecision { get; set; }

    public string? ModelName { get; set; }

    public string? ModelVersion { get; set; }

    public DateTime VerifiedAt { get; set; }

    public string? ReviewerComment { get; set; }

    public virtual Cheque Cheque { get; set; } = null!;

    public virtual User? ReviewedByUser { get; set; }

    public virtual ICollection<SignatureComparison> SignatureComparisons { get; set; } = new List<SignatureComparison>();
}
