using System;
using System.Collections.Generic;

namespace ChequeVerification.Web.Models.Entities;

public partial class Cheque
{
    public int ChequeId { get; set; }

    public int CustomerId { get; set; }

    public int ImportedByUserId { get; set; }

    public string ChequeNumber { get; set; } = null!;

    public decimal? Amount { get; set; }

    public DateOnly? IssueDate { get; set; }

    public string ImagePath { get; set; } = null!;

    public byte Status { get; set; }

    public DateTime UploadedAt { get; set; }

    public virtual Customer Customer { get; set; } = null!;

    public virtual ExtractedSignature? ExtractedSignature { get; set; }

    public virtual User ImportedByUser { get; set; } = null!;

    public virtual ICollection<VerificationResult> VerificationResults { get; set; } = new List<VerificationResult>();
}
