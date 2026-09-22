namespace ChequeVerification.Web.ViewModels.Customers;

public class CustomerListItemViewModel
{
    public int CustomerId { get; set; }
    public string CustomerNumber { get; set; } = string.Empty;
    public string FullName { get; set; } = string.Empty;
    public string AccountNumber { get; set; } = string.Empty;
    public string? Email { get; set; }
    public string? Phone { get; set; }
    public DateTime CreatedAt { get; set; }
    public int ChequeCount { get; set; }
    public int ReferenceSignatureCount { get; set; }
}