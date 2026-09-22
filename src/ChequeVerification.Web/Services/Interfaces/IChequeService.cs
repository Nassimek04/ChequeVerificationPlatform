using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.ViewModels.Cheques;
using Microsoft.AspNetCore.Mvc.Rendering;

namespace ChequeVerification.Web.Services.Interfaces;

public interface IChequeService
{
    Task<ChequeListViewModel> GetPagedAsync(string? searchTerm, int page, int pageSize, CancellationToken cancellationToken = default);
    Task<ChequeDetailsViewModel?> GetDetailsAsync(int id, CancellationToken cancellationToken = default);
    Task<IEnumerable<SelectListItem>> GetCustomerOptionsAsync(CancellationToken cancellationToken = default);
    Task<string> GenerateChequeNumberAsync(CancellationToken cancellationToken = default);
    Task<Cheque> CreateAsync(ChequeCreateViewModel model, int userId, string webRootPath, CancellationToken cancellationToken = default);
}