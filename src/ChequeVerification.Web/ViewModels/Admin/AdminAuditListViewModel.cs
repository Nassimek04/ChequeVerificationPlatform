namespace ChequeVerification.Web.ViewModels.Admin;

public class AdminAuditItemViewModel
{
    public long AuditLogId { get; set; }
    public DateTime CreatedAt { get; set; }
    public string? UserName { get; set; }
    public string Action { get; set; } = string.Empty;
    public string EntityName { get; set; } = string.Empty;
    public int? EntityId { get; set; }
    public string? Description { get; set; }
}

public class AdminAuditListViewModel
{
    public List<AdminAuditItemViewModel> Items { get; set; } = new();
    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 20;
    public long TotalCount { get; set; }
    public int TotalPages => PageSize <= 0 ? 0 : (int)Math.Ceiling(TotalCount / (double)PageSize);
}
