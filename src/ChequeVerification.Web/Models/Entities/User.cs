using System;
using System.Collections.Generic;

namespace ChequeVerification.Web.Models.Entities;

public partial class User
{
    public int UserId { get; set; }

    public int RoleId { get; set; }

    public string FullName { get; set; } = null!;

    public string Email { get; set; } = null!;

    public string PasswordHash { get; set; } = null!;

    public byte Status { get; set; }

    public DateTime CreatedAt { get; set; }

    public DateTime? LastLogin { get; set; }

    public virtual ICollection<AuditLog> AuditLogs { get; set; } = new List<AuditLog>();

    public virtual ICollection<Cheque> Cheques { get; set; } = new List<Cheque>();

    public virtual Role Role { get; set; } = null!;

    public virtual ICollection<VerificationResult> VerificationResults { get; set; } = new List<VerificationResult>();
}
