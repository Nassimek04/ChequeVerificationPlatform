using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Tests;

/// <summary>
/// Focused controller-dashboard aggregation tests (read-only service).
/// - "En attente" uses the true reviewable predicate
///   (AD == ControleManuel && FinalDecision == null && Cheque.Status == ControleManuel),
///   identical to the manual-review queue business rule.
/// - "Dossiers finalisés" counts FinalDecision != null (automatic outcomes included).
/// - Final-decision donut semantics incl. the live-DB shape (8/0/29 of 37).
/// </summary>
public class ControllerDashboardTests
{
    private const byte Conforme = 1;
    private const byte NonConforme = 2;
    private const byte ControleManuel = 3;
    private const byte ChequeVerifie = 3;
    private const byte ChequeControleManuel = 4;
    private const byte ChequeRejete = 5;

    private static DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"controller-dashboard-{Guid.NewGuid():N}").Options;

    private static async Task SeedVerificationAsync(
        ChequeVerificationDbContext db,
        string tag,
        byte automaticDecision,
        byte? finalDecision,
        byte chequeStatus,
        int? reviewedByUserId = null)
    {
        var customer = new Customer
        {
            CustomerNumber = $"CUST-CD-{tag}",
            FullName = $"Client {tag}",
            AccountNumber = $"ACC-{tag}"
        };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();

        var cheque = new Cheque
        {
            CustomerId = customer.CustomerId,
            ImportedByUserId = 1,
            ChequeNumber = $"CHQ-CD-{tag}",
            ImagePath = $"cheque-{tag}.png",
            Status = chequeStatus,
            UploadedAt = DateTime.UtcNow
        };
        db.Cheques.Add(cheque);
        await db.SaveChangesAsync();

        db.VerificationResults.Add(new VerificationResult
        {
            ChequeId = cheque.ChequeId,
            SimilarityScore = 0.5m,
            LowerThresholdUsed = 0.6585m,
            UpperThresholdUsed = 0.9150m,
            AutomaticDecision = automaticDecision,
            FinalDecision = finalDecision,
            ModelName = "sig-verif-ai-v5a",
            ModelVersion = "v5a-phase7",
            VerifiedAt = DateTime.UtcNow,
            ReviewedByUserId = reviewedByUserId,
            ReviewerComment = reviewedByUserId.HasValue ? "review comment" : null
        });
        await db.SaveChangesAsync();
    }

    [Fact]
    public async Task EmptyDatabase_ReturnsZeros()
    {
        using var db = new ChequeVerificationDbContext(CreateOptions());
        var vm = await new DashboardService(db).GetControllerDashboardAsync();

        Assert.Equal(0, vm.PendingReviewCount);
        Assert.Equal(0, vm.RecentDecisionsCount);
        Assert.Equal(0, vm.ValidatedCount);
        Assert.Equal(0, vm.RejectedCount);
        Assert.Equal(0, vm.ManualReviewCompletedCount);
        Assert.Empty(vm.LatestVerificationsToReview);
        Assert.Equal(0, vm.ControlDecisions.Total);
    }

    [Fact]
    public async Task PendingCount_UsesTrueReviewablePredicate()
    {
        using var db = new ChequeVerificationDbContext(CreateOptions());

        // 2 genuinely reviewable dossiers.
        await SeedVerificationAsync(db, "rv1", ControleManuel, null, ChequeControleManuel);
        await SeedVerificationAsync(db, "rv2", ControleManuel, null, ChequeControleManuel);
        // Same decisions but stale cheque status: NOT reviewable (queue would refuse it).
        await SeedVerificationAsync(db, "stale", ControleManuel, null, ChequeVerifie);
        // Automatic outcomes: finalized, never pending.
        await SeedVerificationAsync(db, "auto-c", Conforme, Conforme, ChequeVerifie);
        await SeedVerificationAsync(db, "auto-nc", NonConforme, NonConforme, ChequeRejete);
        // Human-reviewed manual case: finalized + validated, not pending.
        await SeedVerificationAsync(db, "human-c", ControleManuel, Conforme, ChequeVerifie, reviewedByUserId: 7);

        var vm = await new DashboardService(db).GetControllerDashboardAsync();

        Assert.Equal(2, vm.PendingReviewCount);
        Assert.Equal(3, vm.RecentDecisionsCount);
        Assert.Equal(2, vm.ValidatedCount);
        Assert.Equal(1, vm.RejectedCount);
        Assert.Equal(1, vm.ManualReviewCompletedCount);
        Assert.Equal(2, vm.LatestVerificationsToReview.Count);
        Assert.All(vm.LatestVerificationsToReview,
            v => Assert.Null(v.FinalDecision));
        // Pending + finalized partition the whole set: no row counted twice, none dropped.
        Assert.Equal(5, vm.PendingReviewCount + vm.RecentDecisionsCount);
    }

    [Fact]
    public async Task Finalized_IncludesAutomaticDecisions_AndManualCompletedCountIsHumanOnly()
    {
        using var db = new ChequeVerificationDbContext(CreateOptions());

        await SeedVerificationAsync(db, "auto-c", Conforme, Conforme, ChequeVerifie);
        await SeedVerificationAsync(db, "human-c", ControleManuel, Conforme, ChequeVerifie, reviewedByUserId: 7);

        var vm = await new DashboardService(db).GetControllerDashboardAsync();

        // Both rows are finalized outcomes, but only one involved a controller.
        Assert.Equal(2, vm.RecentDecisionsCount);
        Assert.Equal(2, vm.ValidatedCount);
        Assert.Equal(1, vm.ManualReviewCompletedCount);
        Assert.Equal(0, vm.PendingReviewCount);
    }

    [Fact]
    public async Task LiveSemantics_FullFixture_MatchesAuditedCounts()
    {
        // Mirrors the audited development-DB shape:
        // 7 auto-C, 27 auto-NC, 1 manual->C, 2 manual->NC, 25 pending manual.
        using var db = new ChequeVerificationDbContext(CreateOptions());

        for (var i = 0; i < 7; i++)
            await SeedVerificationAsync(db, $"ac{i}", Conforme, Conforme, ChequeVerifie);
        for (var i = 0; i < 27; i++)
            await SeedVerificationAsync(db, $"anc{i}", NonConforme, NonConforme, ChequeRejete);
        await SeedVerificationAsync(db, "m-c", ControleManuel, Conforme, ChequeVerifie, reviewedByUserId: 7);
        await SeedVerificationAsync(db, "m-nc1", ControleManuel, NonConforme, ChequeRejete, reviewedByUserId: 7);
        await SeedVerificationAsync(db, "m-nc2", ControleManuel, NonConforme, ChequeRejete, reviewedByUserId: 7);
        for (var i = 0; i < 25; i++)
            await SeedVerificationAsync(db, $"pend{i}", ControleManuel, null, ChequeControleManuel);

        var vm = await new DashboardService(db).GetControllerDashboardAsync();

        Assert.Equal(25, vm.PendingReviewCount);
        Assert.Equal(37, vm.RecentDecisionsCount);
        Assert.Equal(8, vm.ValidatedCount);
        Assert.Equal(29, vm.RejectedCount);
        Assert.Equal(3, vm.ManualReviewCompletedCount);
        Assert.Equal(8, vm.ControlDecisions.Conforme);
        Assert.Equal(0, vm.ControlDecisions.ControleManuel);
        Assert.Equal(29, vm.ControlDecisions.NonConforme);
        Assert.Equal(37, vm.ControlDecisions.Total);
        // Pending and finalized partition all 62 rows.
        Assert.Equal(62, vm.PendingReviewCount + vm.RecentDecisionsCount);
    }
}
