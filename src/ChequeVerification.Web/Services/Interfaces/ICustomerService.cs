using ChequeVerification.Web.ViewModels.Customers;

namespace ChequeVerification.Web.Services.Interfaces;

public interface ICustomerService
{
    Task<CustomerListViewModel> GetPagedAsync(string? searchTerm, int page, int pageSize);
    Task<CustomerDetailViewModel?> GetDetailAsync(int id);
}
