using System.Data;
using System.Security.Cryptography;
using System.Text;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Verifications;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace ChequeVerification.Web.Services;

public class VerificationService : IVerificationService
{
    // Cheque statuses (see StatusDisplayHelper / real mapping).
    private const byte ChequeStatusInProcessing = 2;

    // Physical storage of extracted signature crops.
    private const string ExtractedSubFolder = "uploads/signatures/extracted";

    private readonly ChequeVerificationDbContext _db;
    private readonly IVerificationApiClient _verificationApiClient;
    private readonly IWebHostEnvironment _environment;
    private readonly ILogger<VerificationService> _logger;
    private readonly VerificationPolicyOptions _policy;

    public VerificationService(
        ChequeVerificationDbContext db,
        IVerificationApiClient verificationApiClient,
        IWebHostEnvironment environment,
        ILogger<VerificationService> logger,
        IOptions<VerificationPolicyOptions>? policyOptions = null)
    {
        _db = db;
        _verificationApiClient = verificationApiClient;
        _environment = environment;
        _logger = logger;
        _policy = policyOptions?.Value ?? new VerificationPolicyOptions();
    }

    public async Task<IReadOnlyList<VerificationChequeOptionViewModel>> GetAvailableChequesAsync(CancellationToken cancellationToken = default)
    {
        return await _db.Cheques
            .AsNoTracking()
            .Include(c => c.Customer)
            .Where(c => c.Status == ChequeStatus.EnAttente)
            .OrderByDescending(c => c.UploadedAt)
            .Select(c => new VerificationChequeOptionViewModel
            {
                ChequeId = c.ChequeId,
                ChequeNumber = c.ChequeNumber,
                CustomerNumber = c.Customer.CustomerNumber,
                CustomerFullName = c.Customer.FullName,
                AccountNumber = c.Customer.AccountNumber,
                IssueDate = c.IssueDate,
                Amount = c.Amount,
                Status = c.Status
            })
            .ToListAsync(cancellationToken);
    }

    public async Task<VerificationPreparationViewModel?> PrepareVerificationAsync(int chequeId, CancellationToken cancellationToken = default)
    {
        var cheque = await _db.Cheques
            .AsNoTracking()
            .Include(c => c.Customer)
            .FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);

        if (cheque == null)
        {
            return null;
        }

        var activeReferenceSignatures = await _db.ReferenceSignatures
            .AsNoTracking()
            .Where(r => r.CustomerId == cheque.CustomerId && r.IsActive)
            .OrderByDescending(r => r.CreatedAt)
            .Select(r => new VerificationReferenceSignatureViewModel
            {
                ReferenceSignatureId = r.ReferenceSignatureId,
                ImagePath = r.ImagePath,
                CreatedAt = r.CreatedAt,
                IsActive = r.IsActive
            })
            .ToListAsync(cancellationToken);

        var hasExistingVerification = await _db.VerificationResults
            .AsNoTracking()
            .AnyAsync(v => v.ChequeId == chequeId, cancellationToken);

        var extractedSignature = await _db.ExtractedSignatures
            .AsNoTracking()
            .FirstOrDefaultAsync(e => e.ChequeId == chequeId, cancellationToken);

        var blockingReason = DetermineBlockingReason(cheque, activeReferenceSignatures.Count, hasExistingVerification, extractedSignature != null);

        return new VerificationPreparationViewModel
        {
            ChequeId = cheque.ChequeId,
            ChequeNumber = cheque.ChequeNumber,
            CustomerId = cheque.CustomerId,
            CustomerNumber = cheque.Customer.CustomerNumber,
            CustomerFullName = cheque.Customer.FullName,
            AccountNumber = cheque.Customer.AccountNumber,
            Amount = cheque.Amount,
            IssueDate = cheque.IssueDate,
            ImagePath = cheque.ImagePath,
            Status = cheque.Status,
            UploadedAt = cheque.UploadedAt,
            CanStartVerification = blockingReason == null,
            BlockingReason = blockingReason,
            HasExistingVerification = hasExistingVerification,
            ReferenceSignatures = activeReferenceSignatures,
            HasExtractedSignature = extractedSignature != null,
            ExtractedSignatureId = extractedSignature?.ExtractedSignatureId,
            ExtractedSignatureImagePath = extractedSignature?.ImagePath,
            ExtractedSignatureQuality = extractedSignature?.ExtractionConfidence
        };
    }

    private static string? DetermineBlockingReason(Models.Entities.Cheque cheque, int activeSignatureCount, bool hasExistingVerification, bool hasExtractedSignature)
    {
        if (string.IsNullOrWhiteSpace(cheque.ImagePath))
        {
            return "L'image du chèque n'est pas disponible.";
        }

        if (hasExistingVerification)
        {
            return "Ce chèque possède déjà un résultat de vérification.";
        }

        var isExtractionInProgress = cheque.Status == ChequeStatus.EnTraitement && hasExtractedSignature;
        if (cheque.Status != ChequeStatus.EnAttente && !isExtractionInProgress)
        {
            return "Ce chèque n'est pas en attente de vérification.";
        }

        // V5-A controlled prototype requires exactly 5 active references (K5)
        if (activeSignatureCount == 0)
        {
            return "Aucune signature de référence active n'est disponible pour ce client.";
        }

        if (activeSignatureCount != 5)
        {
            return "5 active reference signatures are required for V5 verification.";
        }

        return null;
    }

    public async Task<string?> GetChequeImagePathAsync(int chequeId, CancellationToken cancellationToken = default)
    {
        return await _db.Cheques
            .AsNoTracking()
            .Where(c => c.ChequeId == chequeId)
            .Select(c => c.ImagePath)
            .FirstOrDefaultAsync(cancellationToken);
    }

    public async Task<SignatureExtractionOperationResult> ExtractAndPersistSignatureAsync(
        int chequeId,
        int? userId,
        CancellationToken cancellationToken = default)
    {
        // 1. Load the cheque (tracked: its Status will be updated).
        var cheque = await _db.Cheques
            .Include(c => c.Customer)
            .FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);

        if (cheque == null)
        {
            return Failure("Le chèque demandé est introuvable.");
        }

        // 2. Eligibility checks — all before any FastAPI call or SQL write.
        var eligibility = await CheckExtractionEligibilityAsync(cheque, cancellationToken);
        if (!eligibility.IsEligible)
        {
            return Failure(eligibility.BlockReason!);
        }

        // 3. Call FastAPI with the real cheque image. If unreachable or the
        //    response is invalid, abort without any SQL change or file write.
        var apiResult = await ExtractFromApiAsync(cheque, cancellationToken);
        if (apiResult == null)
        {
            return Failure("Le service d'extraction n'est pas disponible ou n'a pas répondu correctement.");
        }

        // 4. Decode + validate the PNG crop returned by FastAPI.
        byte[] cropBytes;
        try
        {
            cropBytes = Convert.FromBase64String(apiResult.SignatureImageBase64);
        }
        catch (FormatException)
        {
            _logger.LogWarning("FastAPI a retourné un Base64 invalide pour le chèque {ChequeId}.", chequeId);
            return Failure("La signature extraite est invalide (Base64 illisible).");
        }

        if (cropBytes.Length == 0)
        {
            return Failure("La signature extraite est vide.");
        }

        if (!LooksLikePng(cropBytes))
        {
            _logger.LogWarning("Le crop de signature retourné pour {ChequeId} n'a pas un en-tête PNG valide.", chequeId);
            return Failure("La signature extraite n'est pas une image PNG valide.");
        }

        // 5. Persist the crop on disk, then the SQL row in one transaction.
        //    File name is a server-side GUID; never a user-provided value.
        var fileName = $"{Guid.NewGuid():N}.png";
        var relativePath = $"/{ExtractedSubFolder}/{fileName}";

        try
        {
            var fullPath = SaveCropToDisk(cropBytes, fileName);

            // ExtractionConfidence currently receives the technical heuristic
            // "extraction_quality" from the OpenCV V2 pipeline. This is NOT a
            // probability nor a similarity score. It stays in [0;1] — no
            // arbitrary percentage conversion. The DB column is decimal(5,4).
            decimal? confidence = (decimal)Math.Clamp(apiResult.ExtractionQuality, 0.0, 1.0);

            // Transaction only makes sense on a relational provider (SQL Server).
            // In-memory providers (tests) ignore it — behavior stays identical.
            var isRelational = _db.Database.IsRelational();
            var transaction = isRelational
                ? await _db.Database.BeginTransactionAsync(cancellationToken)
                : null;

            try
            {
                var alreadyExtracted = await _db.ExtractedSignatures
                    .AnyAsync(e => e.ChequeId == chequeId, cancellationToken);
                if (alreadyExtracted)
                {
                    if (transaction != null)
                    {
                        await transaction.RollbackAsync(cancellationToken);
                    }
                    DeleteFile(fullPath);
                    return Failure("Une signature a déjà été extraite pour ce chèque.");
                }

                var extracted = new ExtractedSignature
                {
                    ChequeId = chequeId,
                    ImagePath = relativePath,
                    FileHash = ComputeSha256Hex(cropBytes),
                    ExtractionConfidence = confidence,
                    ExtractedAt = DateTime.UtcNow
                };

                _db.ExtractedSignatures.Add(extracted);

                // Workflow not finished yet (comparison is a later step), so the
                // cheque goes to "En traitement" (2), never "Vérifié" (3).
                cheque.Status = ChequeStatusInProcessing;

                _db.AuditLogs.Add(new AuditLog
                {
                    UserId = userId,
                    Action = "EXTRACT_SIGNATURE",
                    EntityName = nameof(ExtractedSignature),
                    EntityId = extracted.ExtractedSignatureId,
                    Description = $"Extraction de la signature du chèque {cheque.ChequeNumber}.",
                    CreatedAt = DateTime.UtcNow
                });

                await _db.SaveChangesAsync(cancellationToken);
                if (transaction != null)
                {
                    await transaction.CommitAsync(cancellationToken);
                }

                return new SignatureExtractionOperationResult
                {
                    Success = true,
                    Message = "Signature extraite avec succès.",
                    ExtractedSignatureId = extracted.ExtractedSignatureId,
                    ImagePath = relativePath,
                    ExtractionQuality = (decimal?)apiResult.ExtractionQuality
                };
            }
            catch (DbUpdateException ex)
            {
                if (transaction != null)
                {
                    await transaction.RollbackAsync(cancellationToken);
                }
                // Unique index UQ_ExtractedSignature_Cheque violated by a
                // concurrent request -> never leave a second file behind.
                DeleteFile(fullPath);
                _logger.LogWarning(ex, "Extraction concurrente détectée pour le chèque {ChequeId}.", chequeId);
                return Failure("Une signature a déjà été extraite pour ce chèque.");
            }
        }
        catch (IOException ex)
        {
            _logger.LogError(ex, "Échec de l'écriture du crop pour le chèque {ChequeId}.", chequeId);
            return Failure("Impossible d'enregistrer le fichier de la signature extraite.");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex, "Échec de la persistance de la signature extraite pour le chèque {ChequeId}.", chequeId);
            return Failure("Une erreur est survenue lors de la persistance de la signature.");
        }
    }

    public async Task<SignatureExtractionOperationResult> ReExtractAndPersistSignatureAsync(
        int chequeId,
        int? userId,
        CancellationToken cancellationToken = default)
    {
        // 1. Load the cheque (tracked: its row may be updated).
        var cheque = await _db.Cheques
            .Include(c => c.Customer)
            .FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);

        if (cheque == null)
        {
            return Failure("Le chèque demandé est introuvable.");
        }

        // 2. Eligibility — an existing ExtractedSignature is required, a
        //    VerificationResult blocks the operation. No silent fallback to
        //    first extraction.
        var eligibility = await CheckReExtractionEligibilityAsync(cheque, cancellationToken);
        if (!eligibility.IsEligible)
        {
            return Failure(eligibility.BlockReason!);
        }

        // 3. Load the existing row (tracked, updated in place — never a new row).
        var extracted = await _db.ExtractedSignatures
            .FirstOrDefaultAsync(e => e.ChequeId == chequeId, cancellationToken);

        if (extracted == null)
        {
            return Failure("Aucune signature n'a été extraite pour ce chèque. Exécutez d'abord « Extraire la signature ».");
        }

        var oldImagePath = extracted.ImagePath;
        var oldPhysicalPath = ResolvePhysicalPath(oldImagePath);

        // 4. Call FastAPI with the real cheque image. If unreachable or the
        //    response is invalid, abort without any SQL change or file write.
        var apiResult = await ExtractFromApiAsync(cheque, cancellationToken);
        if (apiResult == null)
        {
            return Failure("Le service d'extraction n'est pas disponible ou n'a pas répondu correctement.");
        }

        // 5. Decode + validate the PNG crop returned by FastAPI.
        byte[] cropBytes;
        try
        {
            cropBytes = Convert.FromBase64String(apiResult.SignatureImageBase64);
        }
        catch (FormatException)
        {
            _logger.LogWarning("FastAPI a retourné un Base64 invalide lors de la ré-extraction du chèque {ChequeId}.", chequeId);
            return Failure("La signature extraite est invalide (Base64 illisible).");
        }

        if (cropBytes.Length == 0)
        {
            return Failure("La signature extraite est vide.");
        }

        if (!LooksLikePng(cropBytes))
        {
            _logger.LogWarning("Le crop de signature retourné pour {ChequeId} n'a pas un en-tête PNG valide (ré-extraction).", chequeId);
            return Failure("La signature extraite n'est pas une image PNG valide.");
        }

        // 6. Persist the NEW crop on disk FIRST (never overwrite the old file).
        //    Server-side GUID, never a user-provided value.
        var fileName = $"{Guid.NewGuid():N}.png";
        var newRelativePath = $"/{ExtractedSubFolder}/{fileName}";

        string newFullPath;
        try
        {
            newFullPath = SaveCropToDisk(cropBytes, fileName);
        }
        catch (IOException ex)
        {
            _logger.LogError(ex, "Échec de l'écriture du nouveau crop pour le chèque {ChequeId}.", chequeId);
            return Failure("Impossible d'enregistrer le fichier de la signature ré-extraite.");
        }

        // 7. SHA-256 over the bytes actually saved + confidence.
        var newHash = ComputeSha256Hex(cropBytes);
        decimal? confidence = (decimal)Math.Clamp(apiResult.ExtractionQuality, 0.0, 1.0);

        // 8. Update the EXISTING row in one transaction. The old physical crop
        //    is only deleted AFTER the commit succeeds.
        var isRelational = _db.Database.IsRelational();
        var transaction = isRelational
            ? await _db.Database.BeginTransactionAsync(cancellationToken)
            : null;

        try
        {
            // Guard against a concurrent verification result landing between
            // eligibility and the update.
            var hasVerificationNow = await _db.VerificationResults
                .AsNoTracking()
                .AnyAsync(v => v.ChequeId == chequeId, cancellationToken);
            if (hasVerificationNow)
            {
                if (transaction != null)
                {
                    await transaction.RollbackAsync(cancellationToken);
                }
                DeleteFile(newFullPath);
                return Failure("Ce chèque possède déjà un résultat de vérification.");
            }

            // Same tracked instance (identity map); re-query documents intent.
            var current = await _db.ExtractedSignatures
                .FirstOrDefaultAsync(e => e.ChequeId == chequeId, cancellationToken);

            if (current == null)
            {
                if (transaction != null)
                {
                    await transaction.RollbackAsync(cancellationToken);
                }
                DeleteFile(newFullPath);
                return Failure("Aucune signature n'a été extraite pour ce chèque. Exécutez d'abord « Extraire la signature ».");
            }

            extracted = current;
            oldImagePath = extracted.ImagePath;
            oldPhysicalPath = ResolvePhysicalPath(oldImagePath);

            extracted.ImagePath = newRelativePath;
            extracted.FileHash = newHash;
            extracted.ExtractionConfidence = confidence;
            extracted.ExtractedAt = DateTime.UtcNow;

            // Workflow semantics preserved: same Status (never Verified), no
            // new status values, no verification result.
            _db.AuditLogs.Add(new AuditLog
            {
                UserId = userId,
                Action = "REEXTRACT_SIGNATURE",
                EntityName = nameof(ExtractedSignature),
                EntityId = extracted.ExtractedSignatureId,
                Description = $"Ré-extraction de la signature du chèque {cheque.ChequeNumber}.",
                CreatedAt = DateTime.UtcNow
            });

            await _db.SaveChangesAsync(cancellationToken);
            if (transaction != null)
            {
                await transaction.CommitAsync(cancellationToken);
            }
        }
        catch (DbUpdateException ex)
        {
            if (transaction != null)
            {
                await transaction.RollbackAsync(cancellationToken);
            }
            DeleteFile(newFullPath);
            _logger.LogWarning(ex, "Échec SQL lors de la ré-extraction du chèque {ChequeId}.", chequeId);
            return Failure("Une erreur est survenue lors de la ré-extraction de la signature.");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            if (transaction != null)
            {
                await transaction.RollbackAsync(cancellationToken);
            }
            DeleteFile(newFullPath);
            _logger.LogError(ex, "Échec de la persistance de la signature ré-extraite pour le chèque {ChequeId}.", chequeId);
            return Failure("Une erreur est survenue lors de la ré-extraction de la signature.");
        }

        // 9. AFTER the commit: best-effort deletion of the old physical crop.
        //    A failure here must NOT roll back the committed DB update.
        if (!string.IsNullOrEmpty(oldPhysicalPath))
        {
            TryDeleteOldExtractedFile(oldPhysicalPath, chequeId);
        }

        return new SignatureExtractionOperationResult
        {
            Success = true,
            Message = "Signature ré-extraite avec succès.",
            ExtractedSignatureId = extracted.ExtractedSignatureId,
            ImagePath = newRelativePath,
            ExtractionQuality = (decimal?)apiResult.ExtractionQuality
        };
    }

    private async Task<EligibilityResult> CheckReExtractionEligibilityAsync(Cheque cheque, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(cheque.ImagePath))
        {
            return new EligibilityResult(false, "L'image du chèque n'est pas disponible.");
        }

        if (cheque.Customer == null)
        {
            return new EligibilityResult(false, "Le client associé au chèque est introuvable.");
        }

        if (!ResolveChequeImageExists(cheque.ImagePath))
        {
            return new EligibilityResult(false, "Le fichier image du chèque est introuvable sur le serveur.");
        }

        var hasActiveReference = await _db.ReferenceSignatures
            .AsNoTracking()
            .AnyAsync(r => r.CustomerId == cheque.CustomerId && r.IsActive, cancellationToken);
        if (!hasActiveReference)
        {
            return new EligibilityResult(false, "Aucune signature de référence active n'est disponible pour ce client.");
        }

        var hasExtracted = await _db.ExtractedSignatures
            .AsNoTracking()
            .AnyAsync(e => e.ChequeId == cheque.ChequeId, cancellationToken);
        if (!hasExtracted)
        {
            return new EligibilityResult(false, "Aucune signature n'a été extraite pour ce chèque. Exécutez d'abord « Extraire la signature ».");
        }

        var hasVerificationResult = await _db.VerificationResults
            .AsNoTracking()
            .AnyAsync(v => v.ChequeId == cheque.ChequeId, cancellationToken);
        if (hasVerificationResult)
        {
            return new EligibilityResult(false, "Ce chèque possède déjà un résultat de vérification.");
        }

        return new EligibilityResult(true, null);
    }

    private void TryDeleteOldExtractedFile(string physicalPath, int chequeId)
    {
        if (!System.IO.File.Exists(physicalPath))
        {
            return;
        }

        try
        {
            System.IO.File.Delete(physicalPath);
        }
        catch (IOException ex)
        {
            _logger.LogWarning(ex, "Impossible de supprimer l'ancien fichier de signature extraite après ré-extraction du chèque {ChequeId}.", chequeId);
        }
        catch (UnauthorizedAccessException ex)
        {
            _logger.LogWarning(ex, "Accès refusé à l'ancien fichier de signature extraite après ré-extraction du chèque {ChequeId}.", chequeId);
        }
    }

    private async Task<EligibilityResult> CheckExtractionEligibilityAsync(Cheque cheque, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(cheque.ImagePath))
        {
            return new EligibilityResult(false, "L'image du chèque n'est pas disponible.");
        }

        if (cheque.Customer == null)
        {
            return new EligibilityResult(false, "Le client associé au chèque est introuvable.");
        }

        if (!ResolveChequeImageExists(cheque.ImagePath))
        {
            return new EligibilityResult(false, "Le fichier image du chèque est introuvable sur le serveur.");
        }

        var hasActiveReference = await _db.ReferenceSignatures
            .AsNoTracking()
            .AnyAsync(r => r.CustomerId == cheque.CustomerId && r.IsActive, cancellationToken);
        if (!hasActiveReference)
        {
            return new EligibilityResult(false, "Aucune signature de référence active n'est disponible pour ce client.");
        }

        var alreadyExtracted = await _db.ExtractedSignatures
            .AsNoTracking()
            .AnyAsync(e => e.ChequeId == cheque.ChequeId, cancellationToken);
        if (alreadyExtracted)
        {
            return new EligibilityResult(false, "Une signature a déjà été extraite pour ce chèque.");
        }

        var hasVerificationResult = await _db.VerificationResults
            .AsNoTracking()
            .AnyAsync(v => v.ChequeId == cheque.ChequeId, cancellationToken);
        if (hasVerificationResult)
        {
            return new EligibilityResult(false, "Ce chèque possède déjà un résultat de vérification.");
        }

        // Fresh automatic starts require Status=1. Kept after the duplicate
        // guards so a double-click reports the pre-existing extraction.
        if (cheque.Status != ChequeStatus.EnAttente)
        {
            return new EligibilityResult(false, "Ce chèque n'est pas en attente de vérification.");
        }

        return new EligibilityResult(true, null);
    }

    private async Task<SignatureExtractionResponseDto?> ExtractFromApiAsync(Cheque cheque, CancellationToken cancellationToken)
    {
        var physicalPath = ResolvePhysicalPath(cheque.ImagePath);
        if (physicalPath == null || !System.IO.File.Exists(physicalPath))
        {
            return null;
        }

        var extension = Path.GetExtension(physicalPath).ToLowerInvariant();
        var contentType = extension switch
        {
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            _ => "application/octet-stream"
        };

        await using var stream = new FileStream(physicalPath, FileMode.Open, FileAccess.Read, FileShare.Read);

        var result = await _verificationApiClient.ExtractSignatureAsync(
            stream,
            Path.GetFileName(physicalPath),
            contentType,
            cancellationToken);

        if (result == null || !result.Success || string.IsNullOrEmpty(result.SignatureImageBase64))
        {
            return null;
        }

        return result;
    }

    private string SaveCropToDisk(byte[] cropBytes, string fileName)
    {
        var root = _environment.WebRootPath;
        var directory = Path.Combine(root, ExtractedSubFolder.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(directory);

        var fullPath = Path.Combine(directory, fileName);
        System.IO.File.WriteAllBytes(fullPath, cropBytes);
        return fullPath;
    }

    private static void DeleteFile(string fullPath)
    {
        try
        {
            if (System.IO.File.Exists(fullPath))
            {
                System.IO.File.Delete(fullPath);
            }
        }
        catch (IOException ex)
        {
            // Logged by the caller; never rethrow to avoid masking the original error.
            System.Diagnostics.Debug.WriteLine(ex.Message);
        }
    }

    private string? ResolvePhysicalPath(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            return null;
        }

        if (imagePath.StartsWith("http://") || imagePath.StartsWith("https://"))
        {
            return null;
        }

        try
        {
            return imagePath.StartsWith("/") || imagePath.StartsWith("\\")
                ? Path.Combine(_environment.WebRootPath, imagePath.TrimStart('/', '\\').Replace('/', Path.DirectorySeparatorChar))
                : imagePath;
        }
        catch
        {
            return null;
        }
    }

    private bool ResolveChequeImageExists(string imagePath)
    {
        var physical = ResolvePhysicalPath(imagePath);
        return physical != null && System.IO.File.Exists(physical);
    }

    private static bool LooksLikePng(byte[] bytes)
    {
        // PNG magic: 89 50 4E 47 0D 0A 1A 0A
        byte[] magic = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
        if (bytes.Length < magic.Length)
        {
            return false;
        }

        for (var i = 0; i < magic.Length; i++)
        {
            if (bytes[i] != magic[i])
            {
                return false;
            }
        }

        return true;
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

    public async Task<SignatureComparisonOperationResult> CompareSignaturesWithReferencesAsync(
        int chequeId,
        CancellationToken cancellationToken = default)
    {
        // Read-only step: no SQL write, no Status change, no audit entry.
        var cheque = await _db.Cheques
            .AsNoTracking()
            .Include(c => c.Customer)
            .FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);

        if (cheque == null)
        {
            return ComparisonFailure("Le chèque demandé est introuvable.");
        }

        var extractedSignature = await _db.ExtractedSignatures
            .AsNoTracking()
            .FirstOrDefaultAsync(e => e.ChequeId == chequeId, cancellationToken);

        if (extractedSignature == null)
        {
            return ComparisonFailure("Aucune signature n'a été extraite pour ce chèque.");
        }

        // 1. Read the extracted signature once (reused for every reference).
        var extractedPhysicalPath = ResolvePhysicalPath(extractedSignature.ImagePath);
        if (extractedPhysicalPath == null || !System.IO.File.Exists(extractedPhysicalPath))
        {
            _logger.LogWarning("Fichier de signature extraite introuvable pour le chèque {ChequeId}.", chequeId);
            return ComparisonFailure("Le fichier de la signature extraite est introuvable sur le serveur.");
        }

        var extractedBytes = await System.IO.File.ReadAllBytesAsync(extractedPhysicalPath, cancellationToken);
        var extractedExtension = Path.GetExtension(extractedPhysicalPath).ToLowerInvariant();
        var extractedContentType = ContentTypeFor(extractedExtension);

        var references = await _db.ReferenceSignatures
            .AsNoTracking()
            .Where(r => r.CustomerId == cheque.CustomerId && r.IsActive)
            .OrderByDescending(r => r.CreatedAt)
            .ToListAsync(cancellationToken);

        if (references.Count == 0)
        {
            return ComparisonFailure("Aucune signature de référence active n'est disponible pour ce client.");
        }

        var comparisons = new List<VerificationReferenceComparisonViewModel>();
        var comparedCount = 0;

        foreach (var reference in references)
        {
            var comparison = new VerificationReferenceComparisonViewModel
            {
                ReferenceSignatureId = reference.ReferenceSignatureId,
                IsAvailable = false,
                StatusMessage = "Fichier introuvable sur le serveur."
            };

            var referencePath = ResolvePhysicalPath(reference.ImagePath);
            if (referencePath == null || !System.IO.File.Exists(referencePath))
            {
                _logger.LogWarning(
                    "Fichier de référence {ReferenceId} introuvable pour la comparaison du chèque {ChequeId}.",
                    reference.ReferenceSignatureId,
                    chequeId);
                comparisons.Add(comparison);
                continue;
            }

            var referenceBytes = await System.IO.File.ReadAllBytesAsync(referencePath, cancellationToken);
            var referenceExtension = Path.GetExtension(referencePath).ToLowerInvariant();

            using var extractedStream = new MemoryStream(extractedBytes);
            using var referenceStream = new MemoryStream(referenceBytes);

            var result = await _verificationApiClient.CompareSignaturesAsync(
                extractedStream,
                Path.GetFileName(extractedPhysicalPath),
                extractedContentType,
                referenceStream,
                Path.GetFileName(referencePath),
                ContentTypeFor(referenceExtension),
                cancellationToken);

            if (result == null || !result.Success)
            {
                comparison.StatusMessage = "Service de comparaison indisponible ou réponse invalide.";
                comparisons.Add(comparison);
                continue;
            }

            comparison.IsAvailable = true;
            comparison.StatusMessage = null;
            comparison.Score = result.SimilarityScore;
            comparison.Method = result.Method;
            comparison.Version = result.Version;
            comparisons.Add(comparison);
            comparedCount++;
        }

        if (comparedCount == 0)
        {
            return new SignatureComparisonOperationResult
            {
                Success = false,
                Message = "Le service de comparaison est indisponible ou n'a pas répondu pour les signatures de référence.",
                Comparisons = comparisons
            };
        }

        var best = comparisons
            .Where(c => c.IsAvailable && c.Score.HasValue)
            .OrderByDescending(c => c.Score!.Value)
            .First();

        return new SignatureComparisonOperationResult
        {
            Success = true,
            Message = "Comparaison expérimentale effectuée.",
            Comparisons = comparisons,
            BestReferenceSignatureId = best.ReferenceSignatureId,
            BestSimilarityScore = best.Score,
            Method = best.Method,
            Version = best.Version
        };
    }

    // AI V2 comparison — fully independent from the OpenCV comparison above:
    // no score fusion, no threshold, no conformity decision and NO persistence
    // (no SQL write, no Status change, no audit entry, no file write). The raw
    // cosine similarity is returned in memory only.
    public async Task<SignatureAiComparisonOperationResult> CompareAiSignaturesWithReferencesAsync(
        int chequeId,
        CancellationToken cancellationToken = default)
    {
        var cheque = await _db.Cheques
            .AsNoTracking()
            .Include(c => c.Customer)
            .FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);

        if (cheque == null)
        {
            return AiComparisonFailure("Le chèque demandé est introuvable.");
        }

        var extractedSignature = await _db.ExtractedSignatures
            .AsNoTracking()
            .FirstOrDefaultAsync(e => e.ChequeId == chequeId, cancellationToken);

        if (extractedSignature == null)
        {
            return AiComparisonFailure("Aucune signature n'a été extraite pour ce chèque.");
        }

        var extractedPhysicalPath = ResolvePhysicalPath(extractedSignature.ImagePath);
        if (extractedPhysicalPath == null || !System.IO.File.Exists(extractedPhysicalPath))
        {
            _logger.LogWarning("Fichier de signature extraite introuvable pour la comparaison IA du chèque {ChequeId}.", chequeId);
            return AiComparisonFailure("Le fichier de la signature extraite est introuvable sur le serveur.");
        }

        var extractedBytes = await System.IO.File.ReadAllBytesAsync(extractedPhysicalPath, cancellationToken);
        var extractedContentType = ContentTypeFor(Path.GetExtension(extractedPhysicalPath).ToLowerInvariant());

        var references = await _db.ReferenceSignatures
            .AsNoTracking()
            .Where(r => r.CustomerId == cheque.CustomerId && r.IsActive)
            .OrderByDescending(r => r.CreatedAt)
            .ToListAsync(cancellationToken);

        if (references.Count == 0)
        {
            return AiComparisonFailure("Aucune signature de référence active n'est disponible pour ce client.");
        }

        var comparisons = new List<VerificationReferenceAiComparisonViewModel>();
        var comparedCount = 0;

        foreach (var reference in references)
        {
            var comparison = new VerificationReferenceAiComparisonViewModel
            {
                ReferenceSignatureId = reference.ReferenceSignatureId,
                IsAvailable = false
            };

            var referencePath = ResolvePhysicalPath(reference.ImagePath);
            if (referencePath == null || !System.IO.File.Exists(referencePath))
            {
                _logger.LogWarning(
                    "Fichier de référence {ReferenceId} introuvable pour la comparaison IA du chèque {ChequeId}.",
                    reference.ReferenceSignatureId,
                    chequeId);
                comparison.StatusMessage = "Fichier introuvable sur le serveur.";
                comparisons.Add(comparison);
                continue;
            }

            var referenceBytes = await System.IO.File.ReadAllBytesAsync(referencePath, cancellationToken);

            using var extractedStream = new MemoryStream(extractedBytes);
            using var referenceStream = new MemoryStream(referenceBytes);

            var result = await _verificationApiClient.CompareSignaturesAiAsync(
                extractedStream,
                Path.GetFileName(extractedPhysicalPath),
                extractedContentType,
                referenceStream,
                Path.GetFileName(referencePath),
                ContentTypeFor(Path.GetExtension(referencePath).ToLowerInvariant()),
                cancellationToken);

            // null: API unreachable / non-structured error. Success=false:
            // structured controlled unavailability (AI disabled, checkpoint or
            // PyTorch missing) — the server-provided reason is displayed.
            if (result == null)
            {
                comparison.StatusMessage = "Service de comparaison IA indisponible ou réponse invalide.";
                comparisons.Add(comparison);
                continue;
            }

            if (!result.Success)
            {
                comparison.StatusMessage = string.IsNullOrWhiteSpace(result.Message)
                    ? "Comparaison IA indisponible."
                    : $"Comparaison IA indisponible : {result.Message}";
                comparisons.Add(comparison);
                continue;
            }

            comparison.IsAvailable = true;
            comparison.StatusMessage = null;
            comparison.Score = result.SimilarityScore;
            comparison.Method = result.Method;
            comparison.Version = result.Version;
            comparison.Model = result.Model;
            comparison.Device = result.Device;
            comparisons.Add(comparison);
            comparedCount++;
        }

        var activeCount = references.Count;
        var unavailableCount = activeCount - comparedCount;
        var isAggregationAvailable = comparedCount > 0;
        double? meanRawScore = null;
        if (isAggregationAvailable)
        {
            var validScores = comparisons
                .Where(c => c.IsAvailable && c.Score.HasValue)
                .Select(c => c.Score!.Value)
                .ToList();
            if (validScores.Count > 0)
            {
                meanRawScore = validScores.Average();
            }
            else
            {
                isAggregationAvailable = false;
            }
        }

        if (comparedCount == 0)
        {
            return new SignatureAiComparisonOperationResult
            {
                Success = false,
                Message = "La comparaison IA est indisponible pour toutes les signatures de référence.",
                Comparisons = comparisons,
                MeanRawScore = null,
                ActiveReferenceCount = activeCount,
                ComparedReferenceCount = comparedCount,
                UnavailableReferenceCount = unavailableCount,
                IsAggregationAvailable = false
            };
        }

        return new SignatureAiComparisonOperationResult
        {
            Success = true,
            Message = $"Comparaison IA effectuée pour {comparedCount}/{references.Count} référence(s).",
            Comparisons = comparisons,
            MeanRawScore = meanRawScore,
            ActiveReferenceCount = activeCount,
            ComparedReferenceCount = comparedCount,
            UnavailableReferenceCount = unavailableCount,
            IsAggregationAvailable = isAggregationAvailable
        };
    }

    public async Task<ChequeOcrOperationResult> OcrChequeAsync(
        int chequeId,
        CancellationToken cancellationToken = default)
    {
        // Read-only OCR flow: load cheque, resolve image, call FastAPI, return result.
        // No DB mutation, no persistence.
        var cheque = await _db.Cheques
            .AsNoTracking()
            .FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);

        if (cheque == null)
        {
            return OcrFailure("Le chèque demandé est introuvable.");
        }

        if (string.IsNullOrWhiteSpace(cheque.ImagePath))
        {
            return OcrFailure("L'image du chèque n'est pas disponible.");
        }

        var physicalPath = ResolvePhysicalPath(cheque.ImagePath);
        if (physicalPath == null || !System.IO.File.Exists(physicalPath))
        {
            _logger.LogWarning("Fichier image du chèque {ChequeId} introuvable pour OCR.", chequeId);
            return OcrFailure("Le fichier image du chèque est introuvable sur le serveur.");
        }

        var extension = Path.GetExtension(physicalPath).ToLowerInvariant();
        var contentType = ContentTypeFor(extension);

        // Guard path traversal: already handled by ResolvePhysicalPath, but reject URLs
        if (cheque.ImagePath.StartsWith("http://") || cheque.ImagePath.StartsWith("https://"))
        {
            return OcrFailure("Chemin d'image invalide.");
        }

        await using var stream = new FileStream(physicalPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        var result = await _verificationApiClient.OcrChequeAsync(
            stream,
            Path.GetFileName(physicalPath),
            contentType,
            cancellationToken);

        if (result == null)
        {
            return OcrFailure("Le service OCR est indisponible ou n'a pas répondu correctement.");
        }

        if (!result.Success)
        {
            // Structured 503 with message from FastAPI
            var msg = string.IsNullOrWhiteSpace(result.Message) ? "Service OCR indisponible." : result.Message;
            return new ChequeOcrOperationResult
            {
                Success = false,
                Message = msg,
                FullText = result.FullText ?? string.Empty,
                Lines = new List<VerificationOcrLineViewModel>(),
                Fields = new VerificationOcrFieldsViewModel(),
                ProcessingMs = result.ProcessingMs,
                Lang = result.Lang ?? string.Empty,
                Device = result.Device ?? string.Empty
            };
        }

        return new ChequeOcrOperationResult
        {
            Success = true,
            Message = string.IsNullOrWhiteSpace(result.Message) ? "OCR effectuée avec succès." : result.Message,
            FullText = result.FullText ?? string.Empty,
            Lines = result.Lines?.Select(l => new VerificationOcrLineViewModel
            {
                Text = l.Text ?? string.Empty,
                Confidence = l.Confidence,
                Box = l.Box ?? new List<List<double>>()
            }).ToList() ?? new List<VerificationOcrLineViewModel>(),
            Fields = new VerificationOcrFieldsViewModel
            {
                ChequeNumber = result.Fields?.ChequeNumber,
                Date = result.Fields?.Date,
                AmountText = result.Fields?.AmountText,
                AmountNumeric = result.Fields?.AmountNumeric,
                AccountNumber = result.Fields?.AccountNumber
            },
            ProcessingMs = result.ProcessingMs,
            Lang = result.Lang ?? string.Empty,
            Device = result.Device ?? string.Empty
        };
    }

    private static ChequeOcrOperationResult OcrFailure(string message)
    {
        return new ChequeOcrOperationResult { Success = false, Message = message };
    }

    public async Task<VerificationSignatureDebugViewModel> GetExtractionDiagnosticAsync(
        int chequeId,
        CancellationToken cancellationToken = default)
    {
        // Read-only diagnostic: resolve the persisted cheque image, rerun the
        // existing V2.4 debug endpoint, map the real DTO. No DB mutation.
        var model = new VerificationSignatureDebugViewModel { ChequeId = chequeId };

        var cheque = await _db.Cheques
            .AsNoTracking()
            .FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);
        if (cheque == null || string.IsNullOrWhiteSpace(cheque.ImagePath))
        {
            model.Success = false;
            model.ErrorMessage = "L'image du chèque n'est pas disponible.";
            return model;
        }

        var physicalPath = ResolvePhysicalPath(cheque.ImagePath);
        if (physicalPath == null || !System.IO.File.Exists(physicalPath))
        {
            model.Success = false;
            model.ErrorMessage = "Le fichier image du chèque est introuvable sur le serveur.";
            return model;
        }

        var extension = Path.GetExtension(physicalPath).ToLowerInvariant();
        await using var stream = new FileStream(physicalPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        var result = await _verificationApiClient.DebugSignatureExtractionAsync(
            stream, Path.GetFileName(physicalPath), ContentTypeFor(extension), cancellationToken);
        if (result == null)
        {
            model.Success = false;
            model.ErrorMessage = "Le diagnostic d'extraction est indisponible.";
            return model;
        }

        MapDebugDto(result, model);
        return model;
    }

    /// <summary>
    /// Maps the real V2.4 debug DTO onto the existing diagnostic view model.
    /// Same evidence as the Create diagnostic flow; no value is invented.
    /// </summary>
    private static void MapDebugDto(SignatureDebugResponseDto result, VerificationSignatureDebugViewModel model)
    {
        model.Success = result.Success;
        model.PipelineVersion = result.ExtractionPipelineVersion;
        model.OriginalWidth = result.OriginalWidth;
        model.OriginalHeight = result.OriginalHeight;
        model.RoiX = result.CandidateRoi?.X;
        model.RoiY = result.CandidateRoi?.Y;
        model.RoiWidth = result.CandidateRoi?.Width;
        model.RoiHeight = result.CandidateRoi?.Height;
        model.MicrBandX = result.MicrBand?.X;
        model.MicrBandY = result.MicrBand?.Y;
        model.MicrBandWidth = result.MicrBand?.Width;
        model.MicrBandHeight = result.MicrBand?.Height;
        model.BboxX = result.SignatureBbox?.X;
        model.BboxY = result.SignatureBbox?.Y;
        model.BboxWidth = result.SignatureBbox?.Width;
        model.BboxHeight = result.SignatureBbox?.Height;
        model.TotalComponentCount = result.TotalComponentCount;
        model.RetainedComponentCount = result.RetainedComponentCount;
        model.RejectedComponentCount = result.RejectedComponentCount;
        model.MicrRejectedCount = result.MicrRejectedCount;
        model.GroupCount = result.GroupCount;
        model.SelectedGroupIndex = result.SelectedGroupIndex;
        model.SelectionReason = result.SelectionReason;
        model.RejectedComponentReasons = result.RejectedComponentReasons;
        model.Groups = result.Groups
            .Select(g => new VerificationDebugGroupViewModel
            {
                Index = g.Index,
                ComponentCount = g.ComponentCount,
                BboxX = g.Bbox?.X,
                BboxY = g.Bbox?.Y,
                BboxWidth = g.Bbox?.Width,
                BboxHeight = g.Bbox?.Height,
                InkArea = g.InkArea,
                Score = g.Score,
                Selected = g.Selected
            })
            .ToList();
        model.ExtractionQuality = result.ExtractionQuality;
        model.Message = result.Message;
        model.DominantComponentX = result.DominantComponentBbox?.X;
        model.DominantComponentY = result.DominantComponentBbox?.Y;
        model.DominantComponentWidth = result.DominantComponentBbox?.Width;
        model.DominantComponentHeight = result.DominantComponentBbox?.Height;
        model.DominantComponentInk = result.DominantComponentInk;
        model.DominantComponentScore = result.DominantComponentScore;
        model.RefinedComponentCount = result.RefinedComponentCount;
        model.CoreComponentIndices = result.CoreComponentIndices;
        model.DiscardedFromSelectedGroupIndices = result.DiscardedFromSelectedGroupIndices;
        model.RefinedBboxX = result.RefinedSignatureBbox?.X;
        model.RefinedBboxY = result.RefinedSignatureBbox?.Y;
        model.RefinedBboxWidth = result.RefinedSignatureBbox?.Width;
        model.RefinedBboxHeight = result.RefinedSignatureBbox?.Height;
        model.RefinementReason = result.RefinementReason;
        model.DirectionalAcceptedIndices = result.DirectionalAcceptedIndices;
        model.DirectionalAcceptDetails = result.DirectionalAcceptDetails
            .Select(d => new VerificationDirectionalAcceptDetailViewModel
            {
                ComponentIndex = d.ComponentIndex,
                AnchorIndex = d.AnchorIndex,
                HGap = d.HGap,
                VGap = d.VGap,
                InkDistance = d.InkDistance,
                HGapLimit = d.HGapLimit,
                InkDistanceLimit = d.InkDistanceLimit,
                Height = d.Height
            })
            .ToList();
        model.CompletenessScore = result.CompletenessScore;
        model.CompletenessRefinedInk = result.CompletenessRefinedInk;
        model.CompletenessReferenceInk = result.CompletenessReferenceInk;
        model.QualityBase = result.QualityBase;
        model.QualityCompletenessFactor = result.QualityCompletenessFactor;
        model.LocalizationMode = string.IsNullOrWhiteSpace(result.LocalizationMode) ? "roi" : result.LocalizationMode;
        model.LocalizationLabel = string.IsNullOrWhiteSpace(result.LocalizationLabel)
            ? (model.LocalizationMode == "global_fallback" ? "Recherche globale de secours" : "ROI principale")
            : result.LocalizationLabel;
        model.RoiFailureReason = result.RoiFailureReason;
        model.FallbackCandidateCount = result.FallbackCandidateCount;
        model.FallbackSelectedIndex = result.FallbackSelectedIndex;
        model.FallbackSelectedScore = result.FallbackSelectedScore;
        model.FallbackSelectionReason = result.FallbackSelectionReason;
        model.FinalCropWidth = result.FinalCropWidth;
        model.FinalCropHeight = result.FinalCropHeight;
        model.FallbackRejectedCandidates = result.FallbackRejectedCandidates
            .Select(c => new VerificationFallbackRejectedCandidateViewModel
            {
                Index = c.Index,
                Reason = c.Reason,
                BboxX = c.Bbox?.X,
                BboxY = c.Bbox?.Y,
                BboxWidth = c.Bbox?.Width,
                BboxHeight = c.Bbox?.Height,
            })
            .ToList();
        model.FallbackCandidates = result.FallbackCandidates
            .Select(c => new VerificationFallbackCandidateViewModel
            {
                Index = c.Index,
                BboxX = c.Bbox?.X,
                BboxY = c.Bbox?.Y,
                BboxWidth = c.Bbox?.Width,
                BboxHeight = c.Bbox?.Height,
                Width = c.Width,
                Height = c.Height,
                Aspect = c.Aspect,
                TrueDensity = c.TrueDensity,
                ComponentCount = c.ComponentCount,
                InkRelative = c.InkRelative,
                VerticalExtent = c.VerticalExtent,
                Score = c.Score,
                Status = c.Status,
                Selected = c.Selected,
            })
            .ToList();
        model.PrimaryRoiX = result.PrimaryRoi?.X;
        model.PrimaryRoiY = result.PrimaryRoi?.Y;
        model.PrimaryRoiWidth = result.PrimaryRoi?.Width;
        model.PrimaryRoiHeight = result.PrimaryRoi?.Height;
        model.RoiCandidateFound = result.RoiCandidateFound;
        model.FallbackExecuted = result.FallbackExecuted;
        model.CropCompleteness = result.CropCompleteness is null
            ? null
            : new VerificationCropCompletenessViewModel
            {
                InitialBboxX = result.CropCompleteness.InitialBbox?.X,
                InitialBboxY = result.CropCompleteness.InitialBbox?.Y,
                InitialBboxWidth = result.CropCompleteness.InitialBbox?.Width,
                InitialBboxHeight = result.CropCompleteness.InitialBbox?.Height,
                FinalBboxX = result.CropCompleteness.FinalBbox?.X,
                FinalBboxY = result.CropCompleteness.FinalBbox?.Y,
                FinalBboxWidth = result.CropCompleteness.FinalBbox?.Width,
                FinalBboxHeight = result.CropCompleteness.FinalBbox?.Height,
                InitialComponentCount = result.CropCompleteness.InitialComponentCount,
                RecoveredComponentCount = result.CropCompleteness.RecoveredComponentCount,
                RecoveredLeft = result.CropCompleteness.RecoveredLeft,
                RecoveredRight = result.CropCompleteness.RecoveredRight,
                RecoveredOther = result.CropCompleteness.RecoveredOther,
                Iterations = result.CropCompleteness.Iterations,
                ExpansionRatio = result.CropCompleteness.ExpansionRatio,
                Status = result.CropCompleteness.Status,
                RecoveredStrokes = result.CropCompleteness.RecoveredStrokes
                    .Select(s => new VerificationRecoveredStrokeViewModel
                    {
                        Direction = s.Direction,
                        BboxX = s.Bbox?.X,
                        BboxY = s.Bbox?.Y,
                        BboxWidth = s.Bbox?.Width,
                        BboxHeight = s.Bbox?.Height,
                        HGap = s.HGap,
                        VGap = s.VGap,
                        Reason = s.Reason,
                    })
                    .ToList(),
            };

        if (!string.IsNullOrEmpty(result.OriginalWithRoiBase64))
            model.OriginalWithRoiDataUri = $"data:image/png;base64,{result.OriginalWithRoiBase64}";
        if (!string.IsNullOrEmpty(result.RoiImageBase64))
            model.RoiImageDataUri = $"data:image/png;base64,{result.RoiImageBase64}";
        if (!string.IsNullOrEmpty(result.MaskImageBase64))
            model.MaskImageDataUri = $"data:image/png;base64,{result.MaskImageBase64}";
        if (!string.IsNullOrEmpty(result.ComponentsAllBase64))
            model.ComponentsAllDataUri = $"data:image/png;base64,{result.ComponentsAllBase64}";
        if (!string.IsNullOrEmpty(result.ComponentsRejectedBase64))
            model.ComponentsRejectedDataUri = $"data:image/png;base64,{result.ComponentsRejectedBase64}";
        if (!string.IsNullOrEmpty(result.ComponentsImageBase64))
            model.ComponentsImageDataUri = $"data:image/png;base64,{result.ComponentsImageBase64}";
        if (!string.IsNullOrEmpty(result.GroupImageBase64))
            model.GroupImageDataUri = $"data:image/png;base64,{result.GroupImageBase64}";
        if (!string.IsNullOrEmpty(result.GroupsImageBase64))
            model.GroupsImageDataUri = $"data:image/png;base64,{result.GroupsImageBase64}";
        if (!string.IsNullOrEmpty(result.RefinedGroupImageBase64))
            model.RefinedGroupImageDataUri = $"data:image/png;base64,{result.RefinedGroupImageBase64}";
        if (!string.IsNullOrEmpty(result.FallbackCandidatesImageBase64))
            model.FallbackCandidatesDataUri = $"data:image/png;base64,{result.FallbackCandidatesImageBase64}";
        if (!string.IsNullOrEmpty(result.SignatureImageBase64))
            model.SignatureImageDataUri = $"data:image/png;base64,{result.SignatureImageBase64}";
    }

    private static string ContentTypeFor(string extension) => extension switch
    {
        ".png" => "image/png",
        ".jpg" or ".jpeg" => "image/jpeg",
        _ => "application/octet-stream"
    };

    private static SignatureComparisonOperationResult ComparisonFailure(string message)
    {
        return new SignatureComparisonOperationResult { Success = false, Message = message };
    }

    private static SignatureAiComparisonOperationResult AiComparisonFailure(string message)
    {
        return new SignatureAiComparisonOperationResult { Success = false, Message = message };
    }

    private static SignatureExtractionOperationResult Failure(string message)
    {
        return new SignatureExtractionOperationResult { Success = false, Message = message };
    }

    public async Task<LaunchVerificationOperationResult> VerifyAutomaticallyAsync(int chequeId, int currentUserId, CancellationToken cancellationToken = default)
    {
        // Automatic flow guard: only Status=1 starts; Status=2 with persisted
        // extraction may finish. Completed/manual/rejected/error are rejected
        // before any extraction or AI call. Reuses existing pipeline below.
        var cheque = await _db.Cheques
            .AsNoTracking()
            .FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);
        if (cheque == null)
            return new LaunchVerificationOperationResult { Success = false, Message = "Le chèque demandé est introuvable." };

        var hasExtracted = await _db.ExtractedSignatures
            .AsNoTracking()
            .AnyAsync(e => e.ChequeId == chequeId, cancellationToken);
        var isFreshStart = cheque.Status == ChequeStatus.EnAttente;
        var isFinishInProgress = cheque.Status == ChequeStatus.EnTraitement && hasExtracted;
        if (!isFreshStart && !isFinishInProgress)
            return new LaunchVerificationOperationResult { Success = false, Message = "Ce chèque n'est pas éligible à une nouvelle vérification." };

        var hasExisting = await _db.VerificationResults
            .AsNoTracking()
            .AnyAsync(v => v.ChequeId == chequeId, cancellationToken);
        if (hasExisting)
            return new LaunchVerificationOperationResult { Success = false, Message = "Ce chèque a déjà été vérifié." };

        // Step 1 (reused): OpenCV V2.4 extraction + persistence. Skipped only
        // when an in-progress cheque already has its persisted extraction.
        if (!hasExtracted)
        {
            var extraction = await ExtractAndPersistSignatureAsync(chequeId, currentUserId, cancellationToken);
            if (!extraction.Success)
                return new LaunchVerificationOperationResult { Success = false, Message = extraction.Message ?? "L'extraction automatique de la signature a échoué." };
        }

        // Step 2 (reused): V5-A K=5 comparison + L/U decision + persistence.
        // Manual zone persists FinalDecision=null for the Contrôleur workflow.
        return await LaunchVerificationAsync(chequeId, currentUserId, cancellationToken);
    }

    public async Task<LaunchVerificationOperationResult> LaunchVerificationAsync(int chequeId, int currentUserId, CancellationToken cancellationToken = default)
    {
        // Policy validation 0 <= L < U <=1 — fail safely, do NOT swap
        if (!_policy.IsValid(out var policyError))
        {
            _logger.LogError("Invalid verification policy: {Error}", policyError);
            return new LaunchVerificationOperationResult { Success = false, Message = "Configuration de vérification invalide." };
        }

        // 1. Load/validate data (outside SQL transaction)
        var cheque = await _db.Cheques
            .AsNoTracking()
            .Include(c => c.Customer)
            .FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);
        if (cheque == null)
            return new LaunchVerificationOperationResult { Success = false, Message = "Le chèque demandé est introuvable." };
        if (cheque.Status != ChequeStatus.EnAttente && cheque.Status != ChequeStatus.EnTraitement)
            return new LaunchVerificationOperationResult { Success = false, Message = "Ce chèque n'est pas éligible à une nouvelle vérification." };
        if (string.IsNullOrWhiteSpace(cheque.ImagePath))
            return new LaunchVerificationOperationResult { Success = false, Message = "L'image du chèque n'est pas disponible." };
        if (cheque.Customer == null)
            return new LaunchVerificationOperationResult { Success = false, Message = "Le client associé au chèque est introuvable." };
        if (!ResolveChequeImageExists(cheque.ImagePath))
            return new LaunchVerificationOperationResult { Success = false, Message = "Le fichier image du chèque est introuvable sur le serveur." };
        if (cheque.ImagePath.StartsWith("http://") || cheque.ImagePath.StartsWith("https://"))
            return new LaunchVerificationOperationResult { Success = false, Message = "Chemin d'image invalide." };

        var extracted = await _db.ExtractedSignatures
            .AsNoTracking()
            .FirstOrDefaultAsync(e => e.ChequeId == chequeId, cancellationToken);
        if (extracted == null)
            return new LaunchVerificationOperationResult { Success = false, Message = "Aucune signature n'a été extraite pour ce chèque. Veuillez d'abord extraire la signature." };
        var extractedPhysical = ResolvePhysicalPath(extracted.ImagePath);
        if (extractedPhysical == null || !System.IO.File.Exists(extractedPhysical))
            return new LaunchVerificationOperationResult { Success = false, Message = "Le fichier de la signature extraite est introuvable sur le serveur." };

        var activeRefs = await _db.ReferenceSignatures
            .AsNoTracking()
            .Where(r => r.CustomerId == cheque.CustomerId && r.IsActive)
            .ToListAsync(cancellationToken);
        // V5-A requires exactly 5 (K5) — do not fabricate/duplicate, do not silently select subset
        if (activeRefs.Count == 0)
            return new LaunchVerificationOperationResult { Success = false, Message = "Aucune signature de référence active n'est disponible pour ce client." };
        if (activeRefs.Count != 5)
            return new LaunchVerificationOperationResult { Success = false, Message = activeRefs.Count < 5 ? "5 active reference signatures are required for V5 verification." : "Le nombre de signatures de référence actives dépasse la limite autorisée (5)." };

        var hasExisting = await _db.VerificationResults.AsNoTracking().AnyAsync(v => v.ChequeId == chequeId, cancellationToken);
        if (hasExisting)
            return new LaunchVerificationOperationResult { Success = false, Message = "Ce chèque a déjà été vérifié." };

        // 2. Call AI V5-A outside SQL transaction — K5 mean raw cosine, V5 checkpoint
        var aiResult = await CompareAiSignaturesWithReferencesAsync(chequeId, cancellationToken);
        if (!aiResult.Success || !aiResult.IsAggregationAvailable || !aiResult.MeanRawScore.HasValue)
            return new LaunchVerificationOperationResult { Success = false, Message = aiResult.Message ?? "La comparaison IA est indisponible pour ce chèque." };
        // V5 requires exactly 5 successful references for K5 calibration; do not decide from K<5
        if (aiResult.ActiveReferenceCount != 5 || aiResult.ComparedReferenceCount != 5)
            return new LaunchVerificationOperationResult { Success = false, Message = "5 active reference signatures are required for V5 verification." };

        var meanScore = aiResult.MeanRawScore.Value;
        // Do NOT clamp raw cosine before policy or persistence (task 7)
        // Round only for decimal(5,4) persistence
        var meanDecimal = (decimal)Math.Round(meanScore, 4, MidpointRounding.AwayFromZero);

        // 3. Decision calculation with exact boundaries
        // score <= L => Non conforme (2), score >= U => Conforme (1), else ControleManuel (3)
        byte automaticDecision;
        if (meanScore <= (double)_policy.LowerThreshold)
            automaticDecision = VerificationDecision.NonConforme;
        else if (meanScore >= (double)_policy.UpperThreshold)
            automaticDecision = VerificationDecision.Conforme;
        else
            automaticDecision = VerificationDecision.ControleManuel;

        byte? finalDecision = null;
        if (automaticDecision == VerificationDecision.Conforme)
            finalDecision = VerificationDecision.Conforme;
        else if (automaticDecision == VerificationDecision.NonConforme)
            finalDecision = VerificationDecision.NonConforme;

        // Determine cheque status mapping
        byte newChequeStatus = automaticDecision switch
        {
            VerificationDecision.Conforme => ChequeStatus.Verifie,
            VerificationDecision.NonConforme => ChequeStatus.Rejete,
            VerificationDecision.ControleManuel => ChequeStatus.ControleManuel,
            _ => ChequeStatus.Erreur
        };

        // Identify best match among successful comparisons
        var successful = aiResult.Comparisons.Where(c => c.IsAvailable && c.Score.HasValue).ToList();
        if (successful.Count == 0)
            return new LaunchVerificationOperationResult { Success = false, Message = "Aucune comparaison IA n'a réussi." };
        var best = successful.OrderByDescending(c => c.Score!.Value).ThenBy(c => c.ReferenceSignatureId).First();

        // 4. Short persistence transaction with re-check (Serializable)
        var isRelational = _db.Database.IsRelational();
        var transaction = isRelational
            ? await _db.Database.BeginTransactionAsync(IsolationLevel.Serializable, cancellationToken)
            : null;
        try
        {
            // Re-check duplicate inside transaction
            var existsInside = await _db.VerificationResults.AsNoTracking().AnyAsync(v => v.ChequeId == chequeId, cancellationToken);
            if (existsInside)
            {
                if (transaction != null) await transaction.RollbackAsync(cancellationToken);
                return new LaunchVerificationOperationResult { Success = false, Message = "Ce chèque a déjà été vérifié." };
            }

            // Load cheque tracked for status update
            var chequeTracked = await _db.Cheques.FirstOrDefaultAsync(c => c.ChequeId == chequeId, cancellationToken);
            if (chequeTracked == null)
            {
                if (transaction != null) await transaction.RollbackAsync(cancellationToken);
                return new LaunchVerificationOperationResult { Success = false, Message = "Le chèque demandé est introuvable." };
            }

            var verification = new VerificationResult
            {
                ChequeId = chequeId,
                SimilarityScore = meanDecimal,
                LowerThresholdUsed = _policy.LowerThreshold,
                UpperThresholdUsed = _policy.UpperThreshold,
                AutomaticDecision = automaticDecision,
                FinalDecision = finalDecision,
                ReviewedByUserId = null,
                ReviewerComment = null,
                ModelName = _policy.ModelName,
                ModelVersion = _policy.ModelVersion,
                VerifiedAt = DateTime.UtcNow
            };
            _db.VerificationResults.Add(verification);
            await _db.SaveChangesAsync(cancellationToken); // obtains VerificationId

            foreach (var comp in successful)
            {
                var isBest = comp.ReferenceSignatureId == best.ReferenceSignatureId;
                var simDec = (decimal)Math.Round(comp.Score!.Value, 4, MidpointRounding.AwayFromZero);
                _db.SignatureComparisons.Add(new SignatureComparison
                {
                    VerificationId = verification.VerificationId,
                    ExtractedSignatureId = extracted.ExtractedSignatureId,
                    ReferenceSignatureId = comp.ReferenceSignatureId,
                    SimilarityScore = simDec,
                    IsBestMatch = isBest
                });
            }

            chequeTracked.Status = newChequeStatus;

            _db.AuditLogs.Add(new AuditLog
            {
                UserId = currentUserId,
                Action = "VERIFY_CHEQUE",
                EntityName = nameof(VerificationResult),
                EntityId = verification.VerificationId,
                Description = $"Vérification du chèque {cheque.ChequeNumber}: décision {automaticDecision} (score {meanDecimal:0.0000}), {aiResult.ComparedReferenceCount}/{aiResult.ActiveReferenceCount} références.",
                CreatedAt = DateTime.UtcNow
            });

            await _db.SaveChangesAsync(cancellationToken);
            if (transaction != null) await transaction.CommitAsync(cancellationToken);

            return new LaunchVerificationOperationResult
            {
                Success = true,
                Message = "Vérification effectuée avec succès.",
                VerificationId = verification.VerificationId,
                SimilarityScore = meanDecimal,
                AutomaticDecision = automaticDecision
            };
        }
        catch (DbUpdateException ex)
        {
            if (transaction != null) await transaction.RollbackAsync(cancellationToken);
            _logger.LogWarning(ex, "Échec de persistance de la vérification pour le chèque {ChequeId}.", chequeId);
            return new LaunchVerificationOperationResult { Success = false, Message = "Une erreur est survenue lors de la persistance de la vérification." };
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            if (transaction != null) await transaction.RollbackAsync(cancellationToken);
            _logger.LogError(ex, "Échec inattendu de la vérification pour le chèque {ChequeId}.", chequeId);
            return new LaunchVerificationOperationResult { Success = false, Message = "Une erreur est survenue lors de la vérification." };
        }
    }

    public async Task<VerificationDetailsViewModel?> GetVerificationDetailsAsync(int verificationId, CancellationToken cancellationToken = default)
    {
        var vr = await _db.VerificationResults
            .AsNoTracking()
            .Include(v => v.Cheque).ThenInclude(c => c.Customer)
            .Include(v => v.SignatureComparisons).ThenInclude(sc => sc.ReferenceSignature)
            .Include(v => v.ReviewedByUser)
            .FirstOrDefaultAsync(v => v.VerificationId == verificationId, cancellationToken);
        if (vr == null) return null;
        return await MapToDetailsAsync(vr, cancellationToken);
    }

    public async Task<VerificationDetailsViewModel?> GetVerificationDetailsByChequeAsync(int chequeId, CancellationToken cancellationToken = default)
    {
        var vr = await _db.VerificationResults
            .AsNoTracking()
            .Include(v => v.Cheque).ThenInclude(c => c.Customer)
            .Include(v => v.SignatureComparisons).ThenInclude(sc => sc.ReferenceSignature)
            .Include(v => v.ReviewedByUser)
            .Where(v => v.ChequeId == chequeId)
            .OrderByDescending(v => v.VerifiedAt)
            .FirstOrDefaultAsync(cancellationToken);
        if (vr == null) return null;
        return await MapToDetailsAsync(vr, cancellationToken);
    }

    public async Task<IReadOnlyList<VerificationHistoryItemViewModel>> GetVerificationHistoryAsync(CancellationToken cancellationToken = default)
    {
        // Strictly read-only: AsNoTracking EF query over persisted rows only.
        // No AI/OCR/OpenCV/extraction call. All rows are returned (no
        // per-cheque deduplication) so older records are never discarded.
        return await _db.VerificationResults
            .AsNoTracking()
            .Include(v => v.Cheque).ThenInclude(c => c.Customer)
            .Include(v => v.ReviewedByUser)
            .OrderByDescending(v => v.VerifiedAt)
            .ThenByDescending(v => v.VerificationId)
            .Select(v => new VerificationHistoryItemViewModel
            {
                VerificationId = v.VerificationId,
                ChequeId = v.ChequeId,
                ChequeNumber = v.Cheque.ChequeNumber,
                CustomerId = v.Cheque.CustomerId,
                CustomerNumber = v.Cheque.Customer.CustomerNumber,
                CustomerFullName = v.Cheque.Customer.FullName,
                VerifiedAt = v.VerifiedAt,
                SimilarityScore = v.SimilarityScore,
                LowerThresholdUsed = v.LowerThresholdUsed,
                UpperThresholdUsed = v.UpperThresholdUsed,
                AutomaticDecision = v.AutomaticDecision,
                FinalDecision = v.FinalDecision,
                ModelName = v.ModelName,
                ModelVersion = v.ModelVersion,
                ReviewedByUserId = v.ReviewedByUserId,
                ReviewedByName = v.ReviewedByUser != null ? v.ReviewedByUser.FullName : null,
                ReviewerComment = v.ReviewerComment
            })
            .ToListAsync(cancellationToken);
    }

    private async Task<VerificationDetailsViewModel> MapToDetailsAsync(VerificationResult vr, CancellationToken cancellationToken)
    {
        var activeCount = await _db.ReferenceSignatures.AsNoTracking().CountAsync(r => r.CustomerId == vr.Cheque.CustomerId && r.IsActive, cancellationToken);
        var extracted = await _db.ExtractedSignatures.AsNoTracking().FirstOrDefaultAsync(e => e.ChequeId == vr.ChequeId, cancellationToken);
        var vm = new VerificationDetailsViewModel
        {
            VerificationId = vr.VerificationId,
            ChequeId = vr.ChequeId,
            ChequeNumber = vr.Cheque.ChequeNumber,
            CustomerId = vr.Cheque.CustomerId,
            CustomerNumber = vr.Cheque.Customer.CustomerNumber,
            CustomerFullName = vr.Cheque.Customer.FullName,
            ChequeStatus = vr.Cheque.Status,
            ChequeImagePath = vr.Cheque.ImagePath,
            ChequeImageIsAccessible = ResolveImageIsAccessible(vr.Cheque.ImagePath),
            ExtractedSignatureImagePath = extracted?.ImagePath,
            ExtractedSignatureImageIsAccessible = extracted != null && ResolveImageIsAccessible(extracted.ImagePath),
            ExtractionQuality = extracted?.ExtractionConfidence,
            ExtractedSignatureId = extracted?.ExtractedSignatureId,
            ExtractedAt = extracted?.ExtractedAt,
            SimilarityScore = vr.SimilarityScore,
            LowerThresholdUsed = vr.LowerThresholdUsed,
            UpperThresholdUsed = vr.UpperThresholdUsed,
            AutomaticDecision = vr.AutomaticDecision,
            FinalDecision = vr.FinalDecision,
            ModelName = vr.ModelName,
            ModelVersion = vr.ModelVersion,
            VerifiedAt = vr.VerifiedAt,
            ReviewedByUserId = vr.ReviewedByUserId,
            ReviewedByName = vr.ReviewedByUser?.FullName,
            ReviewerComment = vr.ReviewerComment,
            ActiveReferenceCount = activeCount,
            ComparedReferenceCount = vr.SignatureComparisons.Count
        };
        foreach (var sc in vr.SignatureComparisons.OrderByDescending(s => s.SimilarityScore).ThenBy(s => s.ReferenceSignatureId))
        {
            vm.Comparisons.Add(new VerificationDetailsComparisonViewModel
            {
                ComparisonId = sc.ComparisonId,
                ReferenceSignatureId = sc.ReferenceSignatureId,
                ReferenceImagePath = sc.ReferenceSignature.ImagePath,
                ReferenceImageIsAccessible = ResolveImageIsAccessible(sc.ReferenceSignature.ImagePath),
                SimilarityScore = sc.SimilarityScore,
                IsBestMatch = sc.IsBestMatch,
                ReferenceCreatedAt = sc.ReferenceSignature.CreatedAt
            });
        }
        return vm;
    }

    private bool ResolveImageIsAccessible(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath)) return false;
        if (imagePath.StartsWith("http://") || imagePath.StartsWith("https://")) return false;
        try
        {
            var localPath = imagePath.StartsWith("/") || imagePath.StartsWith("\\")
                ? Path.Combine(_environment.WebRootPath, imagePath.TrimStart('/', '\\').Replace('/', Path.DirectorySeparatorChar))
                : imagePath;
            return System.IO.File.Exists(localPath);
        }
        catch { return false; }
    }

    private sealed record EligibilityResult(bool IsEligible, string? BlockReason);
}
