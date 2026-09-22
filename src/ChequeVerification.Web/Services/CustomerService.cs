using ChequeVerification.Web.Data;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Customers;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Services;

public class CustomerService : ICustomerService
{
    private readonly ChequeVerificationDbContext _db;

    public CustomerService(ChequeVerificationDbContext db)
    {
        _db = db;
    }

    public async Task<CustomerListViewModel> GetPagedAsync(string? searchTerm, int page, int pageSize)
    {
        page = Math.Max(page, 1);
        pageSize = Math.Max(pageSize, 1);

        var query = _db.Customers.AsNoTracking();

        if (!string.IsNullOrWhiteSpace(searchTerm))
        {
            var term = searchTerm.Trim();
            query = query.Where(c =>
                c.CustomerNumber.Contains(term) ||
                c.FullName.Contains(term) ||
                c.AccountNumber.Contains(term) ||
                (c.NationalId != null && c.NationalId.Contains(term)) ||
                (c.Email != null && c.Email.Contains(term)) ||
                (c.Phone != null && c.Phone.Contains(term)));
        }

        var totalCount = await query.CountAsync();

        var customers = await query
            .OrderBy(c => c.CustomerNumber)
            .Skip((page - 1) * pageSize)
            .Take(pageSize)
            .Select(c => new CustomerListItemViewModel
            {
                CustomerId = c.CustomerId,
                CustomerNumber = c.CustomerNumber,
                FullName = c.FullName,
                AccountNumber = c.AccountNumber,
                Email = c.Email,
                Phone = c.Phone,
                CreatedAt = c.CreatedAt,
                ChequeCount = c.Cheques.Count,
                ReferenceSignatureCount = c.ReferenceSignatures.Count
            })
            .ToListAsync();

        return new CustomerListViewModel
        {
            SearchTerm = searchTerm,
            Customers = customers,
            TotalCount = totalCount,
            Page = page,
            PageSize = pageSize
        };
    }

    public async Task<CustomerDetailViewModel?> GetDetailAsync(int id)
    {
        var customer = await _db.Customers
            .AsNoTracking()
            .FirstOrDefaultAsync(c => c.CustomerId == id);

        if (customer == null)
        {
            return null;
        }

        var cheques = await _db.Cheques
            .AsNoTracking()
            .Where(c => c.CustomerId == id)
            .OrderByDescending(c => c.UploadedAt)
            .ToListAsync();

        var referenceSignatures = await _db.ReferenceSignatures
            .AsNoTracking()
            .Where(r => r.CustomerId == id)
            .OrderByDescending(r => r.CreatedAt)
            .ToListAsync();

        return new CustomerDetailViewModel
        {
            Customer = customer,
            ChequeCount = cheques.Count,
            ReferenceSignatureCount = referenceSignatures.Count,
            Cheques = cheques,
            ReferenceSignatures = referenceSignatures
        };
    }
}
