using System;
using System.Collections.Generic;

namespace ChequeVerification.Web.Models.Entities;

public partial class ReferenceSignature
{
    public int ReferenceSignatureId { get; set; }

    public int CustomerId { get; set; }

    public string ImagePath { get; set; } = null!;

    public string? FileHash { get; set; }

    public DateTime CreatedAt { get; set; }

    public bool IsActive { get; set; }

    public virtual Customer Customer { get; set; } = null!;

    public virtual ICollection<SignatureComparison> SignatureComparisons { get; set; } = new List<SignatureComparison>();
}
