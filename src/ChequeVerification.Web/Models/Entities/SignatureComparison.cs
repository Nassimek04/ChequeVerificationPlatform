using System;
using System.Collections.Generic;

namespace ChequeVerification.Web.Models.Entities;

public partial class SignatureComparison
{
    public int ComparisonId { get; set; }

    public int VerificationId { get; set; }

    public int ExtractedSignatureId { get; set; }

    public int ReferenceSignatureId { get; set; }

    public decimal SimilarityScore { get; set; }

    public bool IsBestMatch { get; set; }

    public virtual ExtractedSignature ExtractedSignature { get; set; } = null!;

    public virtual ReferenceSignature ReferenceSignature { get; set; } = null!;

    public virtual VerificationResult Verification { get; set; } = null!;
}
