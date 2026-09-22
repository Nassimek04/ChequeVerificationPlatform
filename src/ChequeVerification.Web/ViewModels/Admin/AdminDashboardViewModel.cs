namespace ChequeVerification.Web.ViewModels.Admin;

public class AdminDashboardViewModel
{
    public int TotalUsers { get; set; }
    public int ActiveUsers { get; set; }
    public int InactiveUsers { get; set; }
    public List<AdminUsersByRoleViewModel> UsersByRole { get; set; } = new();

    public List<AdminRoleViewModel> Roles { get; set; } = new();

    public int TotalReferenceSignatures { get; set; }
    public int ActiveReferenceSignatures { get; set; }

    public decimal LowerThreshold { get; set; }
    public decimal UpperThreshold { get; set; }
    public string ModelName { get; set; } = string.Empty;
    public string ModelVersion { get; set; } = string.Empty;

    public int TotalCustomers { get; set; }
    public int TotalCheques { get; set; }
    public int TotalVerifications { get; set; }
    public int PendingManualReviews { get; set; }

    public long TotalAuditLogs { get; set; }
    public List<AdminAuditItemViewModel> RecentAuditLogs { get; set; } = new();
}

public class AdminUsersByRoleViewModel
{
    public string RoleName { get; set; } = string.Empty;
    public int Count { get; set; }
}

public class AdminRoleViewModel
{
    public int RoleId { get; set; }
    public string Name { get; set; } = string.Empty;
    public string? Description { get; set; }
    public int UserCount { get; set; }
}
