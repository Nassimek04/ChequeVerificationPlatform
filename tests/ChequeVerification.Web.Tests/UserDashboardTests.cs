using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Tests;

/// <summary>
/// Focused User-dashboard KPI tests (read-only service).
/// "Contrôle manuel" means dossiers currently pending manual control,
/// using the same reviewable predicate as the manual-review queue:
/// AD == ControleManuel && FinalDecision == null && Cheque.Status == ControleManuel.
/// Finalized rows (even once-manual ones) must NOT be counted.
/// </summary>
public class UserDashboardTests
{
    private const byte Conforme = 1;
    private const byte NonConforme = 2;
    private const byte ControleManuel = 3;
    private const byte ChequeVerifie = 3;
    private const byte ChequeControleManuel = 4;
    private const byte ChequeRejete = 5;

    private static DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"user-dashboard-{Guid.NewGuid():N}").Options;

    private static async Task SeedVerificationAsync(
        ChequeVerificationDbContext db,
        string tag,
        byte automaticDecision,
        byte? finalDecision,
        byte chequeStatus)
    {
        var customer = new Customer
        {
            CustomerNumber = $"CUST-UD-{tag}",
            FullName = $"Client {tag}",
            AccountNumber = $"ACC-{tag}"
        };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();

        var cheque = new Cheque
        {
            CustomerId = customer.CustomerId,
            ImportedByUserId = 1,
            ChequeNumber = $"CHQ-UD-{tag}",
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
            VerifiedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();
    }

    [Fact]
    public async Task ManualKpi_CountsOnlyCurrentlyPendingReviewable()
    {
        using var db = new ChequeVerificationDbContext(CreateOptions());

        // 1. reviewable manual dossier => counted.
        await SeedVerificationAsync(db, "pending", ControleManuel, null, ChequeControleManuel);
        // 2. manual finalized Conforme => NOT counted.
        await SeedVerificationAsync(db, "fin-c", ControleManuel, Conforme, ChequeVerifie);
        // 3. manual finalized Non conforme => NOT counted.
        await SeedVerificationAsync(db, "fin-nc", ControleManuel, NonConforme, ChequeRejete);
        // 4. auto Conforme => NOT counted.
        await SeedVerificationAsync(db, "auto-c", Conforme, Conforme, ChequeVerifie);
        // 5. auto Non conforme => NOT counted.
        await SeedVerificationAsync(db, "auto-nc", NonConforme, NonConforme, ChequeRejete);
        // Stale status (manual, unfinalized, cheque no longer parked) => NOT counted.
        await SeedVerificationAsync(db, "stale", ControleManuel, null, ChequeVerifie);

        var vm = await new DashboardService(db).GetUserDashboardAsync();

        Assert.Equal(1, vm.ManualReviewCount);
        // Conforme logic unchanged: the two Conforme-finalized rows.
        Assert.Equal(2, vm.ConformeCount);
        Assert.Equal(6, vm.TotalVerifications);
    }

    [Fact]
    public async Task ManualKpi_LiveFixtureShape_EqualsQueueCount()
    {
        // Mirrors the audited development-DB shape:
        // 7 auto-C, 27 auto-NC, 1 manual->C, 2 manual->NC, 25 pending manual.
        using var db = new ChequeVerificationDbContext(CreateOptions());

        for (var i = 0; i < 7; i++)
            await SeedVerificationAsync(db, $"ac{i}", Conforme, Conforme, ChequeVerifie);
        for (var i = 0; i < 27; i++)
            await SeedVerificationAsync(db, $"anc{i}", NonConforme, NonConforme, ChequeRejete);
        await SeedVerificationAsync(db, "m-c", ControleManuel, Conforme, ChequeVerifie);
        await SeedVerificationAsync(db, "m-nc1", ControleManuel, NonConforme, ChequeRejete);
        await SeedVerificationAsync(db, "m-nc2", ControleManuel, NonConforme, ChequeRejete);
        for (var i = 0; i < 25; i++)
            await SeedVerificationAsync(db, $"pend{i}", ControleManuel, null, ChequeControleManuel);

        var vm = await new DashboardService(db).GetUserDashboardAsync();

        Assert.Equal(25, vm.ManualReviewCount);
        Assert.Equal(8, vm.ConformeCount);
        Assert.Equal(62, vm.TotalVerifications);
    }
}
