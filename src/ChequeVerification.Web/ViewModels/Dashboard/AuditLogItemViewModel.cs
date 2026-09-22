namespace ChequeVerification.Web.ViewModels.Dashboard;

public class AuditLogItemViewModel
{
    public long AuditLogId { get; set; }
    public string? UserName { get; set; }
    public string Action { get; set; } = string.Empty;
    public string EntityName { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
}
