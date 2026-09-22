using System;
using System.Collections.Generic;

namespace ChequeVerification.Web.Models.Entities;

public partial class Customer
{
    public int CustomerId { get; set; }

    public string CustomerNumber { get; set; } = null!;

    public string FullName { get; set; } = null!;

    public string AccountNumber { get; set; } = null!;

    public string? NationalId { get; set; }

    public string? Phone { get; set; }

    public string? Email { get; set; }

    public DateTime CreatedAt { get; set; }

    public virtual ICollection<Cheque> Cheques { get; set; } = new List<Cheque>();

    public virtual ICollection<ReferenceSignature> ReferenceSignatures { get; set; } = new List<ReferenceSignature>();
}
