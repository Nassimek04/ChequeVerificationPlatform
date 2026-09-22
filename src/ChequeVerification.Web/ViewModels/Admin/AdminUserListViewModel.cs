namespace ChequeVerification.Web.ViewModels.Admin;

/// <summary>
/// Read-only admin user row. Never contains PasswordHash or secrets.
/// </summary>
public class AdminUserListItemViewModel
{
    public int UserId { get; set; }
    public string FullName { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public string RoleName { get; set; } = string.Empty;
    public byte Status { get; set; }
    public bool IsActive { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? LastLogin { get; set; }
}

public class AdminUserListViewModel
{
    public List<AdminUserListItemViewModel> Items { get; set; } = new();
    public int TotalCount { get; set; }
}
