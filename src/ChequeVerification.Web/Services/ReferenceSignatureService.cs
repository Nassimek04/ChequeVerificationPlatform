using System.Security.Cryptography;
using System.Text;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.ReferenceSignatures;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Services;

public class ReferenceSignatureService : IReferenceSignatureService
{
    private const string UploadsSubFolder = "uploads/signatures/reference";
    private static readonly string[] AllowedExtensions = { ".png", ".jpg", ".jpeg" };
    private static readonly string[] AllowedContentTypes = { "image/png", "image/jpeg" };
    private const int MaxFileSizeBytes = 5 * 1024 * 1024;
    private const int MaxActiveReferences = 5;

    // SECURITY TODO: Reference signatures are currently stored below wwwroot and are
    // directly web-accessible via GUID-obscured filenames. Before production, move
    // sensitive reference images outside public wwwroot and serve them through an
    // authorized controller endpoint (e.g., /ReferenceSignatures/Image/{id}) with
    // role checks, or at minimum enforce no directory listing and audit access.

    private readonly ChequeVerificationDbContext _db;
    private readonly ILogger<ReferenceSignatureService> _logger;

    public ReferenceSignatureService(ChequeVerificationDbContext db, ILogger<ReferenceSignatureService> logger)
    {
        _db = db;
        _logger = logger;
    }

    public async Task<ReferenceSignatureListViewModel> GetPagedAsync(string? searchTerm, int page, int pageSize, CancellationToken cancellationToken = default)
    {
        page = Math.Max(page, 1);
        pageSize = Math.Max(pageSize, 1);

        IQueryable<ReferenceSignature> query = _db.ReferenceSignatures
            .AsNoTracking()
            .Include(r => r.Customer);

        if (!string.IsNullOrWhiteSpace(searchTerm))
        {
            var term = searchTerm.Trim();
            query = query.Where(r =>
                r.Customer.CustomerNumber.Contains(term) ||
                r.Customer.FullName.Contains(term) ||
                r.ImagePath.Contains(term));
        }

        var totalCount = await query.CountAsync(cancellationToken);

        var signatures = await query
            .OrderByDescending(r => r.CreatedAt)
            .Skip((page - 1) * pageSize)
            .Take(pageSize)
            .Select(r => new ReferenceSignatureListItemViewModel
            {
                ReferenceSignatureId = r.ReferenceSignatureId,
                CustomerNumber = r.Customer.CustomerNumber,
                CustomerFullName = r.Customer.FullName,
                ImagePath = r.ImagePath,
                IsActive = r.IsActive,
                CreatedAt = r.CreatedAt
            })
            .ToListAsync(cancellationToken);

        return new ReferenceSignatureListViewModel
        {
            SearchTerm = searchTerm,
            Signatures = signatures,
            TotalCount = totalCount,
            Page = page,
            PageSize = pageSize
        };
    }

    public async Task<ReferenceSignatureDetailsViewModel?> GetDetailsAsync(int id, CancellationToken cancellationToken = default)
    {
        return await _db.ReferenceSignatures
            .AsNoTracking()
            .Include(r => r.Customer)
            .Where(r => r.ReferenceSignatureId == id)
            .Select(r => new ReferenceSignatureDetailsViewModel
            {
                ReferenceSignatureId = r.ReferenceSignatureId,
                CustomerId = r.CustomerId,
                CustomerNumber = r.Customer.CustomerNumber,
                CustomerFullName = r.Customer.FullName,
                ImagePath = r.ImagePath,
                FileHash = r.FileHash,
                IsActive = r.IsActive,
                CreatedAt = r.CreatedAt
            })
            .FirstOrDefaultAsync(cancellationToken);
    }

    public async Task<ReferenceSignatureCreateViewModel?> GetCreateViewModelAsync(int customerId, CancellationToken cancellationToken = default)
    {
        var customer = await _db.Customers
            .AsNoTracking()
            .FirstOrDefaultAsync(c => c.CustomerId == customerId, cancellationToken);

        if (customer == null)
        {
            return null;
        }

        var activeCount = await _db.ReferenceSignatures
            .AsNoTracking()
            .CountAsync(r => r.CustomerId == customerId && r.IsActive, cancellationToken);

        return new ReferenceSignatureCreateViewModel
        {
            CustomerId = customer.CustomerId,
            CustomerNumber = customer.CustomerNumber,
            CustomerFullName = customer.FullName,
            ActiveReferenceCount = activeCount,
            RecommendedTarget = MaxActiveReferences
        };
    }

    public async Task<ReferenceSignatureEnrollmentResult> CreateAsync(int customerId, IFormFile file, int? userId, string webRootPath, CancellationToken cancellationToken = default)
    {
        // 1. Basic null / empty checks.
        if (file == null)
        {
            return Failure("Aucun fichier n'a été fourni.");
        }

        if (file.Length <= 0)
        {
            return Failure("Le fichier est vide.");
        }

        if (file.Length > MaxFileSizeBytes)
        {
            return Failure("La taille du fichier dépasse la limite autorisée de 5 Mo.");
        }

        // 2. Reject absolute paths / URLs / path traversal in original filename.
        var originalFileName = file.FileName ?? string.Empty;
        if (originalFileName.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            originalFileName.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            return Failure("Nom de fichier invalide.");
        }

        if (originalFileName.Contains("..", StringComparison.Ordinal))
        {
            return Failure("Nom de fichier invalide.");
        }

        // 3. Extension validation (from FileName, but not trusted alone).
        var extension = Path.GetExtension(originalFileName).ToLowerInvariant();
        if (string.IsNullOrEmpty(extension) || !AllowedExtensions.Contains(extension))
        {
            return Failure("Format d'image non autorisé. Formats acceptés : .png, .jpg, .jpeg.");
        }

        // 4. ContentType allowlist (additional check, not trusted alone).
        var contentType = file.ContentType?.ToLowerInvariant() ?? string.Empty;
        if (!AllowedContentTypes.Contains(contentType))
        {
            return Failure("Type de contenu non autorisé. Formats acceptés : image/jpeg, image/png.");
        }

        // 5. Customer existence.
        var customerExists = await _db.Customers
            .AsNoTracking()
            .AnyAsync(c => c.CustomerId == customerId, cancellationToken);
        if (!customerExists)
        {
            return Failure("Le client demandé est introuvable.");
        }

        // 6. Max active references check.
        var activeCount = await _db.ReferenceSignatures
            .CountAsync(r => r.CustomerId == customerId && r.IsActive, cancellationToken);
        if (activeCount >= MaxActiveReferences)
        {
            return Failure("Le client possède déjà 5 signatures de référence actives.");
        }

        // 7. Read bytes, magic-byte validation, compute hash.
        byte[] bytes;
        try
        {
            await using var ms = new MemoryStream();
            await file.CopyToAsync(ms, cancellationToken);
            bytes = ms.ToArray();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Échec de la lecture du fichier de signature de référence pour le client {CustomerId}.", customerId);
            return Failure("Impossible de lire le fichier fourni.");
        }

        if (bytes.Length == 0)
        {
            return Failure("Le fichier est vide.");
        }

        if (bytes.Length > MaxFileSizeBytes)
        {
            return Failure("La taille du fichier dépasse la limite autorisée de 5 Mo.");
        }

        if (!HasValidMagicBytes(bytes))
        {
            return Failure("Le contenu du fichier ne correspond pas à une image PNG ou JPEG valide.");
        }

        var hash = ComputeSha256Hex(bytes);

        // 8. Duplicate detection (same customer, same hash).
        var duplicateExists = await _db.ReferenceSignatures
            .AsNoTracking()
            .AnyAsync(r => r.CustomerId == customerId && r.FileHash == hash, cancellationToken);
        if (duplicateExists)
        {
            return Failure("Une signature identique existe déjà pour ce client.");
        }

        // 9. Generate server-side filename and save.
        var fileName = $"{Guid.NewGuid():N}{extension}";
        var relativePath = $"/{UploadsSubFolder}/{fileName}";
        var uploadsDir = Path.Combine(webRootPath, UploadsSubFolder.Replace('/', Path.DirectorySeparatorChar));
        string fullPath;
        try
        {
            Directory.CreateDirectory(uploadsDir);
            fullPath = Path.Combine(uploadsDir, fileName);
            // Use CreateNew to never overwrite.
            await using var fs = new FileStream(fullPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            await fs.WriteAsync(bytes, cancellationToken);
        }
        catch (IOException ex)
        {
            _logger.LogError(ex, "Échec de l'écriture du fichier de référence pour le client {CustomerId}.", customerId);
            return Failure("Impossible d'enregistrer le fichier de la signature de référence.");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex, "Échec de l'écriture du fichier de référence pour le client {CustomerId}.", customerId);
            return Failure("Impossible d'enregistrer le fichier de la signature de référence.");
        }

        // 10. Persist DB row; on failure, delete orphan file.
        var entity = new ReferenceSignature
        {
            CustomerId = customerId,
            ImagePath = relativePath,
            FileHash = hash,
            CreatedAt = DateTime.UtcNow,
            IsActive = true
        };

        try
        {
            _db.ReferenceSignatures.Add(entity);
            await _db.SaveChangesAsync(cancellationToken);
        }
        catch (Exception ex) when (ex is DbUpdateException or InvalidOperationException)
        {
            DeleteFileIfExists(fullPath);
            _logger.LogWarning(ex, "Échec SQL lors de l'enrôlement de la signature pour le client {CustomerId}.", customerId);
            return Failure("Une erreur est survenue lors de l'enregistrement de la signature.");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            DeleteFileIfExists(fullPath);
            _logger.LogError(ex, "Échec de l'enrôlement de la signature pour le client {CustomerId}.", customerId);
            return Failure("Une erreur est survenue lors de l'enregistrement de la signature.");
        }

        // 11. Audit log (best-effort, do not roll back enrollment on audit failure).
        try
        {
            _db.AuditLogs.Add(new AuditLog
            {
                UserId = userId,
                Action = "ENROLL_REFERENCE",
                EntityName = nameof(ReferenceSignature),
                EntityId = entity.ReferenceSignatureId,
                Description = $"Enrôlement de la signature de référence {entity.ReferenceSignatureId} pour le client {customerId}.",
                CreatedAt = DateTime.UtcNow
            });
            await _db.SaveChangesAsync(cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Impossible d'enregistrer l'audit ENROLL_REFERENCE pour la signature {ReferenceSignatureId}.", entity.ReferenceSignatureId);
        }

        return new ReferenceSignatureEnrollmentResult
        {
            Success = true,
            Message = "Signature de référence enrôlée avec succès.",
            ReferenceSignatureId = entity.ReferenceSignatureId,
            ImagePath = relativePath,
            FileHash = hash
        };
    }

    private static bool HasValidMagicBytes(byte[] bytes)
    {
        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if (bytes.Length >= 8 &&
            bytes[0] == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4E && bytes[3] == 0x47 &&
            bytes[4] == 0x0D && bytes[5] == 0x0A && bytes[6] == 0x1A && bytes[7] == 0x0A)
        {
            return true;
        }

        // JPEG: FF D8 FF
        if (bytes.Length >= 3 &&
            bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF)
        {
            return true;
        }

        return false;
    }

    private static string ComputeSha256Hex(byte[] bytes)
    {
        var hash = SHA256.HashData(bytes);
        var sb = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
        {
            sb.Append(b.ToString("x2"));
        }
        return sb.ToString();
    }

    private static void DeleteFileIfExists(string fullPath)
    {
        try
        {
            if (File.Exists(fullPath))
            {
                File.Delete(fullPath);
            }
        }
        catch
        {
            // Best-effort.
        }
    }

    private static ReferenceSignatureEnrollmentResult Failure(string message)
    {
        return new ReferenceSignatureEnrollmentResult { Success = false, Message = message };
    }
}