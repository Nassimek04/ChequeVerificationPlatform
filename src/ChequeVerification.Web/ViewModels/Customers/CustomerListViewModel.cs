namespace ChequeVerification.Web.ViewModels.Customers;

public class CustomerListViewModel
{
    public string? SearchTerm { get; set; }
    public List<CustomerListItemViewModel> Customers { get; set; } = new();
    public int TotalCount { get; set; }
    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 10;
    public int TotalPages => PageSize <= 0 ? 0 : (int)Math.Ceiling(TotalCount / (double)PageSize);
}