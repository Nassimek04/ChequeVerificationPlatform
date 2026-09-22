using ChequeVerification.Web.Services;
using ChequeVerification.Web.ViewModels.ReferenceSignatures;

namespace ChequeVerification.Web.Services.Interfaces;

public interface IReferenceSignatureService
{
    Task<ReferenceSignatureListViewModel> GetPagedAsync(string? searchTerm, int page, int pageSize, CancellationToken cancellationToken = default);
    Task<ReferenceSignatureDetailsViewModel?> GetDetailsAsync(int id, CancellationToken cancellationToken = default);
    Task<ReferenceSignatureEnrollmentResult> CreateAsync(int customerId, IFormFile file, int? userId, string webRootPath, CancellationToken cancellationToken = default);
    Task<ReferenceSignatureCreateViewModel?> GetCreateViewModelAsync(int customerId, CancellationToken cancellationToken = default);
}