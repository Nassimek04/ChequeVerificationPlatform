using System.Globalization;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Dashboard;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Services;

public class DashboardService : IDashboardService
{
    private const int ActivityWindowDays = 14;
    private static readonly CultureInfo Fr = CultureInfo.GetCultureInfo("fr-FR");

    private readonly ChequeVerificationDbContext _db;

    public DashboardService(ChequeVerificationDbContext db)
    {
        _db = db;
    }

    /// <summary>
    /// READ-ONLY analytics over persisted verification records.
    /// Single narrow-column round-trip; grouping done in memory to stay
    /// provider-agnostic. No writes, no recomputation of decisions.
    /// </summary>
    private async Task<List<VerificationAnalyticsRow>> GetVerificationRowsAsync()
    {
        return await _db.VerificationResults
            .AsNoTracking()
            .Select(v => new VerificationAnalyticsRow(
                v.VerifiedAt,
                v.AutomaticDecision,
                v.FinalDecision,
                v.SimilarityScore,
                v.LowerThresholdUsed,
                v.UpperThresholdUsed))
            .ToListAsync();
    }

    private static VerificationActivitySeries BuildActivitySeries(List<VerificationAnalyticsRow> rows)
    {
        var series = new VerificationActivitySeries { WindowDays = ActivityWindowDays };
        var today = DateTime.Today;
        var byDay = rows
            .GroupBy(r => r.VerifiedAt.Date)
            .ToDictionary(g => g.Key, g => g.Count());
        for (var d = ActivityWindowDays - 1; d >= 0; d--)
        {
            var day = today.AddDays(-d);
            series.Labels.Add(day.ToString("dd/MM", Fr));
            series.Counts.Add(byDay.TryGetValue(day, out var c) ? c : 0);
        }
        return series;
    }

    /// <summary>Effective decision = final controller decision, else automatic decision.</summary>
    private static DecisionDistribution BuildEffectiveDecisions(List<VerificationAnalyticsRow> rows)
    {
        var dist = new DecisionDistribution();
        foreach (var r in rows)
        {
            switch (r.FinalDecision ?? r.AutomaticDecision)
            {
                case VerificationDecision.Conforme: dist.Conforme++; break;
                case VerificationDecision.NonConforme: dist.NonConforme++; break;
                case VerificationDecision.ControleManuel: dist.ControleManuel++; break;
            }
        }
        return dist;
    }

    /// <summary>Final controller decisions only (decided cases).</summary>
    private static DecisionDistribution BuildControlDecisions(List<VerificationAnalyticsRow> rows)
    {
        var dist = new DecisionDistribution();
        foreach (var r in rows.Where(r => r.FinalDecision.HasValue))
        {
            switch (r.FinalDecision)
            {
                case VerificationDecision.Conforme: dist.Conforme++; break;
                case VerificationDecision.NonConforme: dist.NonConforme++; break;
                case VerificationDecision.ControleManuel: dist.ControleManuel++; break;
            }
        }
        return dist;
    }

    /// <summary>
    /// Descriptive score zones using each record's OWN persisted thresholds.
    /// Same boundary semantics as the policy (≥ U conforme, ≤ L non conforme).
    /// </summary>
    private static ScoreZoneDistribution BuildScoreZones(List<VerificationAnalyticsRow> rows)
    {
        var zones = new ScoreZoneDistribution();
        foreach (var r in rows)
        {
            if (r.SimilarityScore >= r.UpperThresholdUsed) zones.ZoneConforme++;
            else if (r.SimilarityScore <= r.LowerThresholdUsed) zones.ZoneNonConforme++;
            else zones.ZoneManuelle++;
        }
        return zones;
    }

    private async Task<ChequeStatusDistribution> GetChequePipelineAsync()
    {
        var groups = await _db.Cheques
            .AsNoTracking()
            .GroupBy(c => c.Status)
            .Select(g => new { Status = g.Key, Count = g.Count() })
            .OrderBy(g => g.Status)
            .ToListAsync();
        var dist = new ChequeStatusDistribution();
        foreach (var g in groups)
        {
            dist.Items.Add(new ChequeStatusItem
            {
                Status = g.Status,
                Label = StatusDisplayHelper.ChequeStatusLabel(g.Status),
                Count = g.Count
            });
        }
        return dist;
    }

    private sealed record VerificationAnalyticsRow(
        DateTime VerifiedAt,
        byte AutomaticDecision,
        byte? FinalDecision,
        decimal SimilarityScore,
        decimal LowerThresholdUsed,
        decimal UpperThresholdUsed);

    public async Task<UserDashboardViewModel> GetUserDashboardAsync()
    {
        var latestVerifications = await _db.VerificationResults
            .AsNoTracking()
            .OrderByDescending(v => v.VerifiedAt)
            .Take(5)
            .Select(v => new VerificationItemViewModel
            {
                VerificationId = v.VerificationId,
                ChequeNumber = v.Cheque.ChequeNumber,
                CustomerName = v.Cheque.Customer.FullName,
                SimilarityScore = v.SimilarityScore,
                AutomaticDecision = v.AutomaticDecision,
                FinalDecision = v.FinalDecision,
                VerifiedAt = v.VerifiedAt
            })
            .ToListAsync();

        return new UserDashboardViewModel
        {
            TotalCheques = await _db.Cheques.AsNoTracking().CountAsync(),
            TotalVerifications = await _db.VerificationResults.AsNoTracking().CountAsync(),
            ConformeCount = await _db.VerificationResults.AsNoTracking().CountAsync(v => v.AutomaticDecision == 1 || v.FinalDecision == 1),
            // Currently pending manual control only: same reviewable predicate as the
            // manual-review queue. Rows finalized by a controller (FinalDecision set)
            // are outcomes, not pending dossiers, even if once classified manual.
            ManualReviewCount = await _db.VerificationResults.AsNoTracking().CountAsync(v => v.AutomaticDecision == VerificationDecision.ControleManuel && v.FinalDecision == null && v.Cheque.Status == ChequeStatus.ControleManuel),
            LatestVerifications = latestVerifications,
            Activity = BuildActivitySeries(await GetVerificationRowsAsync())
        };
    }

    public async Task<ControllerDashboardViewModel> GetControllerDashboardAsync()
    {
        // Truly reviewable dossiers only: same business predicate as the
        // manual-review queue (automatic manual decision, no final decision
        // yet, cheque still parked in the manual-control workflow status).
        // A bare FinalDecision == null check would also catch rows that the
        // Examiner workflow would refuse.
        var latestToReview = await _db.VerificationResults
            .AsNoTracking()
            .Where(v => v.AutomaticDecision == VerificationDecision.ControleManuel
                && v.FinalDecision == null
                && v.Cheque.Status == ChequeStatus.ControleManuel)
            .OrderByDescending(v => v.VerifiedAt)
            .Take(5)
            .Select(v => new VerificationItemViewModel
            {
                VerificationId = v.VerificationId,
                ChequeNumber = v.Cheque.ChequeNumber,
                CustomerName = v.Cheque.Customer.FullName,
                SimilarityScore = v.SimilarityScore,
                AutomaticDecision = v.AutomaticDecision,
                FinalDecision = v.FinalDecision,
                VerifiedAt = v.VerifiedAt
            })
            .ToListAsync();

        return new ControllerDashboardViewModel
        {
            PendingReviewCount = await _db.VerificationResults.AsNoTracking().CountAsync(v => v.AutomaticDecision == VerificationDecision.ControleManuel && v.FinalDecision == null && v.Cheque.Status == ChequeStatus.ControleManuel),
            RecentDecisionsCount = await _db.VerificationResults.AsNoTracking().CountAsync(v => v.FinalDecision != null),
            ValidatedCount = await _db.VerificationResults.AsNoTracking().CountAsync(v => v.FinalDecision == 1),
            RejectedCount = await _db.VerificationResults.AsNoTracking().CountAsync(v => v.FinalDecision == 2),
            ManualReviewCompletedCount = await _db.VerificationResults.AsNoTracking().CountAsync(v => v.ReviewedByUserId != null),
            LatestVerificationsToReview = latestToReview,
            ControlDecisions = BuildControlDecisions(await GetVerificationRowsAsync())
        };
    }

    public async Task<AdminDashboardViewModel> GetAdminDashboardAsync()
    {
        var recentActivity = await _db.AuditLogs
            .AsNoTracking()
            .OrderByDescending(a => a.CreatedAt)
            .Take(8)
            .Select(a => new AuditLogItemViewModel
            {
                AuditLogId = a.AuditLogId,
                UserName = a.User != null ? a.User.FullName : null,
                Action = a.Action,
                EntityName = a.EntityName,
                CreatedAt = a.CreatedAt
            })
            .ToListAsync();

        var verificationRows = await GetVerificationRowsAsync();

        return new AdminDashboardViewModel
        {
            TotalUsers = await _db.Users.AsNoTracking().CountAsync(),
            TotalCustomers = await _db.Customers.AsNoTracking().CountAsync(),
            TotalCheques = await _db.Cheques.AsNoTracking().CountAsync(),
            TotalVerifications = await _db.VerificationResults.AsNoTracking().CountAsync(),
            RecentActivity = recentActivity,
            Activity = BuildActivitySeries(verificationRows),
            Decisions = BuildEffectiveDecisions(verificationRows),
            ScoreZones = BuildScoreZones(verificationRows),
            ChequePipeline = await GetChequePipelineAsync(),
            RecentVerifications = await _db.VerificationResults
                .AsNoTracking()
                .OrderByDescending(v => v.VerifiedAt)
                .Take(6)
                .Select(v => new VerificationItemViewModel
                {
                    VerificationId = v.VerificationId,
                    ChequeNumber = v.Cheque.ChequeNumber,
                    CustomerName = v.Cheque.Customer.FullName,
                    SimilarityScore = v.SimilarityScore,
                    AutomaticDecision = v.AutomaticDecision,
                    FinalDecision = v.FinalDecision,
                    VerifiedAt = v.VerifiedAt
                })
                .ToListAsync()
        };
    }
}