using System;
using System.Collections.Generic;

namespace ChequeVerification.Web.Models.Entities;

public partial class ExtractedSignature
{
    public int ExtractedSignatureId { get; set; }

    public int ChequeId { get; set; }

    public string ImagePath { get; set; } = null!;

    public string? FileHash { get; set; }

    public decimal? ExtractionConfidence { get; set; }

    public DateTime ExtractedAt { get; set; }

    public virtual Cheque Cheque { get; set; } = null!;

    public virtual ICollection<SignatureComparison> SignatureComparisons { get; set; } = new List<SignatureComparison>();
}
