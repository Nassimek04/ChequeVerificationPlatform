using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Cheques;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Services;

public class ChequeService : IChequeService
{
    private const string ChequeNumberPrefix = "CHQ-";
    private const string UploadsSubFolder = "uploads/cheques";
    private static readonly string[] AllowedExtensions = { ".jpg", ".jpeg", ".png" };
    private static readonly string[] AllowedContentTypes = { "image/jpeg", "image/png" };
    private const int MaxFileSizeBytes = 10 * 1024 * 1024;

    private readonly ChequeVerificationDbContext _db;
    private readonly ILogger<ChequeService> _logger;

    public ChequeService(ChequeVerificationDbContext db, ILogger<ChequeService> logger)
    {
        _db = db;
        _logger = logger;
    }

    public async Task<ChequeListViewModel> GetPagedAsync(string? searchTerm, int page, int pageSize, CancellationToken cancellationToken = default)
    {
        page = Math.Max(page, 1);
        pageSize = Math.Max(pageSize, 1);

        IQueryable<Cheque> query = _db.Cheques
            .AsNoTracking()
            .Include(c => c.Customer)
            .Include(c => c.ImportedByUser);

        if (!string.IsNullOrWhiteSpace(searchTerm))
        {
            var term = searchTerm.Trim();
            query = query.Where(c =>
                c.ChequeNumber.Contains(term) ||
                c.Customer.CustomerNumber.Contains(term) ||
                c.Customer.FullName.Contains(term) ||
                c.Customer.AccountNumber.Contains(term));
        }

        var totalCount = await query.CountAsync(cancellationToken);

        var cheques = await query
            .OrderByDescending(c => c.UploadedAt)
            .Skip((page - 1) * pageSize)
            .Take(pageSize)
            .Select(c => new ChequeListItemViewModel
            {
                ChequeId = c.ChequeId,
                ChequeNumber = c.ChequeNumber,
                CustomerNumber = c.Customer.CustomerNumber,
                CustomerFullName = c.Customer.FullName,
                AccountNumber = c.Customer.AccountNumber,
                Status = c.Status,
                UploadedAt = c.UploadedAt,
                ImportedByFullName = c.ImportedByUser.FullName
            })
            .ToListAsync(cancellationToken);

        return new ChequeListViewModel
        {
            SearchTerm = searchTerm,
            Cheques = cheques,
            TotalCount = totalCount,
            Page = page,
            PageSize = pageSize
        };
    }

    public async Task<ChequeDetailsViewModel?> GetDetailsAsync(int id, CancellationToken cancellationToken = default)
    {
        return await _db.Cheques
            .AsNoTracking()
            .Include(c => c.Customer)
            .Include(c => c.ImportedByUser)
            .Where(c => c.ChequeId == id)
            .Select(c => new ChequeDetailsViewModel
            {
                ChequeId = c.ChequeId,
                ChequeNumber = c.ChequeNumber,
                CustomerId = c.CustomerId,
                CustomerNumber = c.Customer.CustomerNumber,
                CustomerFullName = c.Customer.FullName,
                AccountNumber = c.Customer.AccountNumber,
                Amount = c.Amount,
                IssueDate = c.IssueDate,
                ImagePath = c.ImagePath,
                Status = c.Status,
                UploadedAt = c.UploadedAt,
                ImportedByFullName = c.ImportedByUser.FullName,
                HasExtractedSignature = c.ExtractedSignature != null,
                HasVerificationResult = c.VerificationResults.Any()
            })
            .FirstOrDefaultAsync(cancellationToken);
    }

    public async Task<IEnumerable<SelectListItem>> GetCustomerOptionsAsync(CancellationToken cancellationToken = default)
    {
        return await _db.Customers
            .AsNoTracking()
            .OrderBy(c => c.CustomerNumber)
            .Select(c => new SelectListItem
            {
                Value = c.CustomerId.ToString(),
                Text = $"{c.CustomerNumber} - {c.FullName}"
            })
            .ToListAsync(cancellationToken);
    }

    public async Task<string> GenerateChequeNumberAsync(CancellationToken cancellationToken = default)
    {
        var last = await _db.Cheques
            .AsNoTracking()
            .OrderByDescending(c => c.ChequeId)
            .Select(c => c.ChequeNumber)
            .FirstOrDefaultAsync(cancellationToken);

        int next = 1;
        if (!string.IsNullOrEmpty(last) && last.StartsWith(ChequeNumberPrefix))
        {
            if (int.TryParse(last[ChequeNumberPrefix.Length..], out var parsed))
            {
                next = parsed + 1;
            }
        }

        return $"{ChequeNumberPrefix}{next:D4}";
    }

    public async Task<Cheque> CreateAsync(ChequeCreateViewModel model, int userId, string webRootPath, CancellationToken cancellationToken = default)
    {
        var imagePath = await SaveImageAsync(model.ImageFile!, webRootPath, cancellationToken);

        var cheque = new Cheque
        {
            CustomerId = model.CustomerId,
            ImportedByUserId = userId,
            ChequeNumber = model.ChequeNumber,
            Amount = model.Amount,
            IssueDate = model.IssueDate,
            ImagePath = imagePath,
            Status = 1,
            UploadedAt = DateTime.UtcNow
        };

        try
        {
            _db.Cheques.Add(cheque);
            await _db.SaveChangesAsync(cancellationToken);
        }
        catch
        {
            DeleteFileIfExists(webRootPath, imagePath);
            throw;
        }

        await LogAuditAsync(userId, "IMPORT_CHEQUE", cheque.ChequeId, $"Import du chèque {cheque.ChequeNumber} pour le client n° {model.CustomerId}", cancellationToken);

        return cheque;
    }

    private async Task<string> SaveImageAsync(IFormFile file, string webRootPath, CancellationToken cancellationToken)
    {
        var extension = Path.GetExtension(file.FileName).ToLowerInvariant();
        if (!AllowedExtensions.Contains(extension))
        {
            throw new InvalidOperationException("Format d'image non autorisé. Formats acceptés : .jpg, .jpeg, .png.");
        }

        if (!AllowedContentTypes.Contains(file.ContentType.ToLowerInvariant()))
        {
            throw new InvalidOperationException("Type de contenu non autorisé. Formats acceptés : image/jpeg, image/png.");
        }

        if (file.Length <= 0 || file.Length > MaxFileSizeBytes)
        {
            throw new InvalidOperationException("Taille du fichier invalide (maximum 10 Mo).");
        }

        var uploadsDir = Path.Combine(webRootPath, UploadsSubFolder.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(uploadsDir);

        var fileName = $"{Guid.NewGuid():N}{extension}";
        var fullPath = Path.Combine(uploadsDir, fileName);

        await using (var stream = new FileStream(fullPath, FileMode.CreateNew))
        {
            await file.CopyToAsync(stream, cancellationToken);
        }

        return $"/{UploadsSubFolder}/{fileName}";
    }

    private void DeleteFileIfExists(string webRootPath, string imagePath)
    {
        try
        {
            if (!imagePath.StartsWith("/"))
            {
                return;
            }

            var relative = imagePath.TrimStart('/').Replace('/', Path.DirectorySeparatorChar);
            var fullPath = Path.Combine(webRootPath, relative);

            if (System.IO.File.Exists(fullPath))
            {
                System.IO.File.Delete(fullPath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Impossible de supprimer le fichier orphelin {ImagePath}", imagePath);
        }
    }

    private async Task LogAuditAsync(int userId, string action, int entityId, string description, CancellationToken cancellationToken)
    {
        try
        {
            _db.AuditLogs.Add(new AuditLog
            {
                UserId = userId,
                Action = action,
                EntityName = nameof(Cheque),
                EntityId = entityId,
                Description = description,
                CreatedAt = DateTime.UtcNow
            });
            await _db.SaveChangesAsync(cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Impossible d'enregistrer l'entrée d'audit pour {Action}", action);
        }
    }
}