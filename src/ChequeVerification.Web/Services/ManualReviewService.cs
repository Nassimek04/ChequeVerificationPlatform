using System.Data;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.ManualReviews;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Services;

public class ManualReviewService : IManualReviewService
{
    private static readonly SemaphoreSlim _inMemoryLock = new(1, 1);

    private readonly ChequeVerificationDbContext _db;
    private readonly IWebHostEnvironment _environment;
    private readonly ILogger<ManualReviewService> _logger;

    public ManualReviewService(
        ChequeVerificationDbContext db,
        IWebHostEnvironment environment,
        ILogger<ManualReviewService> logger)
    {
        _db = db;
        _environment = environment;
        _logger = logger;
    }

    public async Task<IReadOnlyList<ManualReviewQueueItemViewModel>> GetPendingQueueAsync(CancellationToken cancellationToken = default)
    {
        var query = _db.VerificationResults
            .AsNoTracking()
            .Include(v => v.Cheque).ThenInclude(c => c.Customer)
            .Where(v => v.AutomaticDecision == VerificationDecision.ControleManuel
                        && v.FinalDecision == null
                        && v.Cheque.Status == ChequeStatus.ControleManuel)
            .OrderBy(v => v.VerifiedAt);

        var list = await query.Select(v => new ManualReviewQueueItemViewModel
        {
            VerificationId = v.VerificationId,
            ChequeId = v.ChequeId,
            ChequeNumber = v.Cheque.ChequeNumber,
            CustomerId = v.Cheque.CustomerId,
            CustomerNumber = v.Cheque.Customer.CustomerNumber,
            CustomerFullName = v.Cheque.Customer.FullName,
            VerifiedAt = v.VerifiedAt,
            SimilarityScore = v.SimilarityScore,
            ModelName = v.ModelName,
            ModelVersion = v.ModelVersion
        }).ToListAsync(cancellationToken);

        return list;
    }

    public async Task<ManualReviewDetailsViewModel?> GetReviewDetailsAsync(int verificationId, CancellationToken cancellationToken = default)
    {
        var vr = await _db.VerificationResults
            .AsNoTracking()
            .Include(v => v.Cheque).ThenInclude(c => c.Customer)
            .Include(v => v.SignatureComparisons).ThenInclude(sc => sc.ReferenceSignature)
            .Include(v => v.Cheque).ThenInclude(c => c.ExtractedSignature)
            .FirstOrDefaultAsync(v => v.VerificationId == verificationId, cancellationToken);

        if (vr == null) return null;

        // Only pending manual reviews should be reviewable, but we still return details for any verification
        // The caller (controller) will check if it's pending; we return the data regardless for display
        var cheque = vr.Cheque;
        var extracted = await _db.ExtractedSignatures.AsNoTracking()
            .FirstOrDefaultAsync(e => e.ChequeId == cheque.ChequeId, cancellationToken);

        var vm = new ManualReviewDetailsViewModel
        {
            VerificationId = vr.VerificationId,
            ChequeId = vr.ChequeId,
            VerifiedAt = vr.VerifiedAt,
            SimilarityScore = vr.SimilarityScore,
            LowerThresholdUsed = vr.LowerThresholdUsed,
            UpperThresholdUsed = vr.UpperThresholdUsed,
            AutomaticDecision = vr.AutomaticDecision,
            FinalDecision = vr.FinalDecision,
            ModelName = vr.ModelName,
            ModelVersion = vr.ModelVersion,
            ChequeNumber = cheque.ChequeNumber,
            Amount = cheque.Amount,
            IssueDate = cheque.IssueDate,
            ChequeImagePath = cheque.ImagePath,
            ChequeImageIsAccessible = ResolveImageIsAccessible(cheque.ImagePath),
            ChequeStatus = cheque.Status,
            CustomerId = cheque.CustomerId,
            CustomerNumber = cheque.Customer.CustomerNumber,
            CustomerFullName = cheque.Customer.FullName,
            AccountNumber = cheque.Customer.AccountNumber,
            ExtractedSignatureImagePath = extracted?.ImagePath,
            ExtractedSignatureIsAccessible = extracted != null && ResolveImageIsAccessible(extracted.ImagePath),
            ExtractionConfidence = extracted?.ExtractionConfidence,
        };

        foreach (var sc in vr.SignatureComparisons.OrderByDescending(s => s.SimilarityScore).ThenBy(s => s.ReferenceSignatureId))
        {
            vm.References.Add(new ManualReviewReferenceViewModel
            {
                ReferenceSignatureId = sc.ReferenceSignatureId,
                ImagePath = sc.ReferenceSignature.ImagePath,
                ImageIsAccessible = ResolveImageIsAccessible(sc.ReferenceSignature.ImagePath),
                SimilarityScore = sc.SimilarityScore,
                IsBestMatch = sc.IsBestMatch,
                CreatedAt = sc.ReferenceSignature.CreatedAt
            });
        }

        vm.Decision = new ManualReviewDecisionViewModel { VerificationId = vr.VerificationId };

        return vm;
    }

    public async Task<ManualReviewDecisionResult> DecideAsync(int verificationId, byte finalDecision, string reviewerComment, int reviewerUserId, CancellationToken cancellationToken = default)
    {
        if (finalDecision != VerificationDecision.Conforme && finalDecision != VerificationDecision.NonConforme)
        {
            return new ManualReviewDecisionResult { Success = false, Message = "Décision invalide. Choisissez Conforme ou Non conforme." };
        }

        if (string.IsNullOrWhiteSpace(reviewerComment) || reviewerComment.Trim().Length < 5 || reviewerComment.Trim().Length > 1000)
        {
            return new ManualReviewDecisionResult { Success = false, Message = "Un commentaire est requis (5 à 1000 caractères)." };
        }

        reviewerComment = reviewerComment.Trim();

        var isRelational = _db.Database.IsRelational();

        // For InMemory provider, use a global lock to simulate Serializable
        if (!isRelational)
        {
            await _inMemoryLock.WaitAsync(cancellationToken);
        }

        var transaction = isRelational
            ? await _db.Database.BeginTransactionAsync(IsolationLevel.Serializable, cancellationToken)
            : null;

        try
        {
            // Load with tracking inside transaction/lock for concurrency check
            var vr = await _db.VerificationResults
                .Include(v => v.Cheque)
                .FirstOrDefaultAsync(v => v.VerificationId == verificationId, cancellationToken);

            if (vr == null)
            {
                if (transaction != null) await transaction.RollbackAsync(cancellationToken);
                return new ManualReviewDecisionResult { Success = false, Message = "Vérification introuvable." };
            }

            // Validate still pending: AutomaticDecision=3, FinalDecision IS NULL, Cheque.Status=4
            if (vr.AutomaticDecision != VerificationDecision.ControleManuel || vr.FinalDecision != null || vr.Cheque.Status != ChequeStatus.ControleManuel)
            {
                if (transaction != null) await transaction.RollbackAsync(cancellationToken);
                return new ManualReviewDecisionResult { Success = false, Message = "Cette vérification n'est plus en attente de contrôle manuel." };
            }

            // Re-check with fresh query for InMemory to handle race between load and update
            if (!isRelational)
            {
                var fresh = await _db.VerificationResults.AsNoTracking()
                    .FirstOrDefaultAsync(v => v.VerificationId == verificationId, cancellationToken);
                if (fresh == null || fresh.FinalDecision != null || fresh.AutomaticDecision != VerificationDecision.ControleManuel)
                {
                    return new ManualReviewDecisionResult { Success = false, Message = "Cette vérification a déjà été traitée par un autre contrôleur." };
                }
                var freshCheque = await _db.Cheques.AsNoTracking().FirstOrDefaultAsync(c => c.ChequeId == vr.ChequeId, cancellationToken);
                if (freshCheque == null || freshCheque.Status != ChequeStatus.ControleManuel)
                {
                    return new ManualReviewDecisionResult { Success = false, Message = "Le chèque associé n'est plus en attente de contrôle." };
                }
            }

            // Atomic update using ExecuteUpdate would be more robust for race, but we already have tracked entity
            // To handle race, we will attempt update and check rows affected via concurrency
            // Use ExecuteUpdate with condition for true atomicity if available (EF Core 7+)
            // For InMemory provider, ExecuteUpdate is not supported, so we fallback to tracked update + check

            if (isRelational)
            {
                // Use ExecuteUpdate for atomic check - but we already have tracked vr, so we need to ensure no other transaction modified it
                // We will do a conditional update via ExecuteUpdate and then verify
                var rows = await _db.VerificationResults
                    .Where(v => v.VerificationId == verificationId && v.FinalDecision == null && v.AutomaticDecision == VerificationDecision.ControleManuel)
                    .ExecuteUpdateAsync(s => s
                        .SetProperty(v => v.FinalDecision, finalDecision)
                        .SetProperty(v => v.ReviewedByUserId, reviewerUserId)
                        .SetProperty(v => v.ReviewerComment, reviewerComment), cancellationToken);

                if (rows == 0)
                {
                    await transaction!.RollbackAsync(cancellationToken);
                    return new ManualReviewDecisionResult { Success = false, Message = "Cette vérification a déjà été traitée par un autre contrôleur." };
                }

                // Reload to get updated vr for cheque status update
                vr = await _db.VerificationResults.Include(v => v.Cheque).FirstAsync(v => v.VerificationId == verificationId, cancellationToken);
            }
            else
            {
                // InMemory path: check and update via tracked entity
                vr.FinalDecision = finalDecision;
                vr.ReviewedByUserId = reviewerUserId;
                vr.ReviewerComment = reviewerComment;
            }

            // Update Cheque status atomically
            byte newChequeStatus = finalDecision == VerificationDecision.Conforme ? ChequeStatus.Verifie : ChequeStatus.Rejete;

            if (isRelational)
            {
                var chequeRows = await _db.Cheques
                    .Where(c => c.ChequeId == vr.ChequeId && c.Status == ChequeStatus.ControleManuel)
                    .ExecuteUpdateAsync(s => s.SetProperty(c => c.Status, newChequeStatus), cancellationToken);
                if (chequeRows == 0)
                {
                    await transaction!.RollbackAsync(cancellationToken);
                    return new ManualReviewDecisionResult { Success = false, Message = "Le chèque associé n'est plus en attente de contrôle." };
                }
            }
            else
            {
                vr.Cheque.Status = newChequeStatus;
            }

            // Audit
            _db.AuditLogs.Add(new AuditLog
            {
                UserId = reviewerUserId,
                Action = finalDecision == VerificationDecision.Conforme ? "MANUAL_REVIEW_APPROVE" : "MANUAL_REVIEW_REJECT",
                EntityName = nameof(VerificationResult),
                EntityId = verificationId,
                Description = $"Contrôle manuel {(finalDecision == VerificationDecision.Conforme ? "conforme" : "non conforme")} pour la vérification {verificationId} (chèque {vr.ChequeId}) : {reviewerComment}",
                CreatedAt = DateTime.UtcNow
            });

            if (!isRelational)
            {
                // For InMemory, SaveChanges will persist the tracked changes
                await _db.SaveChangesAsync(cancellationToken);
            }
            else
            {
                await _db.SaveChangesAsync(cancellationToken);
                await transaction!.CommitAsync(cancellationToken);
            }

            return new ManualReviewDecisionResult { Success = true, Message = "Décision enregistrée avec succès.", VerificationId = verificationId };
        }
        catch (DbUpdateException ex)
        {
            if (transaction != null) await transaction.RollbackAsync(cancellationToken);
            _logger.LogWarning(ex, "Conflit de concurrence lors de la revue manuelle {VerificationId}", verificationId);
            return new ManualReviewDecisionResult { Success = false, Message = "Conflit de concurrence : cette vérification a déjà été traitée." };
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            if (transaction != null) await transaction.RollbackAsync(cancellationToken);
            _logger.LogError(ex, "Erreur lors de la décision manuelle {VerificationId}", verificationId);
            return new ManualReviewDecisionResult { Success = false, Message = "Une erreur est survenue lors de l'enregistrement de la décision." };
        }
        finally
        {
            if (transaction != null) await transaction.DisposeAsync();
            if (!isRelational)
            {
                _inMemoryLock.Release();
            }
        }
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
}
