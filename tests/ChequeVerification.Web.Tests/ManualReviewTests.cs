using System.Security.Claims;
using ChequeVerification.Web.Controllers;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.ManualReviews;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

public class ManualReviewTests : IDisposable
{
    private readonly string _webRoot = Path.Combine(Path.GetTempPath(), "manual-review-" + Guid.NewGuid().ToString("N"));

    public ManualReviewTests() => Directory.CreateDirectory(_webRoot);
    public void Dispose() { try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { } }

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>().UseInMemoryDatabase($"manual-review-{Guid.NewGuid():N}").Options;
    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());
    private ManualReviewService CreateService(ChequeVerificationDbContext db)
        => new(db, new FakeEnv { WebRootPath = _webRoot }, NullLogger<ManualReviewService>.Instance);

    private async Task<(int verificationId, int chequeId, int customerId)> SeedPendingManualAsync(ChequeVerificationDbContext db, decimal score = 0.7625m, byte autoDecision = 3)
    {
        var customer = new Customer { CustomerNumber = "CUST-MR-1", FullName = "Client MR", AccountNumber = "ACC-MR-1" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        var chequePath = Path.Combine(_webRoot, $"cheque-{Guid.NewGuid():N}.png");
        await File.WriteAllBytesAsync(chequePath, new byte[] { 1, 2, 3 });
        var cheque = new Cheque { CustomerId = customer.CustomerId, ImportedByUserId = 1, ChequeNumber = "CHQ-MR-001", ImagePath = chequePath, Status = 4, UploadedAt = DateTime.UtcNow };
        db.Cheques.Add(cheque);
        await db.SaveChangesAsync();
        var extractedPath = Path.Combine(_webRoot, $"ext-{Guid.NewGuid():N}.png");
        await File.WriteAllBytesAsync(extractedPath, new byte[] { 9, 8, 7 });
        var extracted = new ExtractedSignature { ChequeId = cheque.ChequeId, ImagePath = extractedPath, FileHash = "hash", ExtractionConfidence = 0.8m, ExtractedAt = DateTime.UtcNow };
        db.ExtractedSignatures.Add(extracted);
        await db.SaveChangesAsync();
        for (int i = 1; i <= 5; i++)
        {
            var refPath = Path.Combine(_webRoot, $"ref-{Guid.NewGuid():N}.png");
            await File.WriteAllBytesAsync(refPath, new byte[] { (byte)i });
            db.ReferenceSignatures.Add(new ReferenceSignature { CustomerId = customer.CustomerId, ImagePath = refPath, FileHash = $"hash{i}", CreatedAt = DateTime.UtcNow, IsActive = true });
        }
        await db.SaveChangesAsync();
        var refs = await db.ReferenceSignatures.Where(r => r.CustomerId == customer.CustomerId).ToListAsync();
        var vr = new VerificationResult
        {
            ChequeId = cheque.ChequeId,
            SimilarityScore = score,
            LowerThresholdUsed = 0.6585m,
            UpperThresholdUsed = 0.9150m,
            AutomaticDecision = autoDecision,
            FinalDecision = null,
            ModelName = "sig-verif-ai-v5a",
            ModelVersion = "v5a-phase7",
            VerifiedAt = DateTime.UtcNow,
            ReviewedByUserId = null,
            ReviewerComment = null
        };
        db.VerificationResults.Add(vr);
        await db.SaveChangesAsync();
        foreach (var r in refs)
        {
            db.SignatureComparisons.Add(new SignatureComparison
            {
                VerificationId = vr.VerificationId,
                ExtractedSignatureId = extracted.ExtractedSignatureId,
                ReferenceSignatureId = r.ReferenceSignatureId,
                SimilarityScore = 0.5m + (r.ReferenceSignatureId % 5) * 0.01m,
                IsBestMatch = r.ReferenceSignatureId == refs.First().ReferenceSignatureId
            });
        }
        await db.SaveChangesAsync();
        return (vr.VerificationId, cheque.ChequeId, customer.CustomerId);
    }

    private async Task<int> SeedNonPendingAsync(ChequeVerificationDbContext db, byte autoDecision, byte? finalDecision, byte chequeStatus)
    {
        var customer = new Customer { CustomerNumber = $"CUST-NP-{Guid.NewGuid():N}", FullName = "Client NP", AccountNumber = "ACC-NP" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        var chequePath = Path.Combine(_webRoot, $"cheque-np-{Guid.NewGuid():N}.png");
        await File.WriteAllBytesAsync(chequePath, new byte[] { 1 });
        var cheque = new Cheque { CustomerId = customer.CustomerId, ImportedByUserId = 1, ChequeNumber = $"CHQ-NP-{Guid.NewGuid():N}", ImagePath = chequePath, Status = chequeStatus, UploadedAt = DateTime.UtcNow };
        db.Cheques.Add(cheque);
        await db.SaveChangesAsync();
        var extractedPath = Path.Combine(_webRoot, $"ext-np-{Guid.NewGuid():N}.png");
        await File.WriteAllBytesAsync(extractedPath, new byte[] { 9 });
        db.ExtractedSignatures.Add(new ExtractedSignature { ChequeId = cheque.ChequeId, ImagePath = extractedPath, FileHash = "h", ExtractionConfidence = 0.5m, ExtractedAt = DateTime.UtcNow });
        await db.SaveChangesAsync();
        for (int i = 0; i < 5; i++)
        {
            var p = Path.Combine(_webRoot, $"ref-np-{Guid.NewGuid():N}.png");
            await File.WriteAllBytesAsync(p, new byte[] { 1 });
            db.ReferenceSignatures.Add(new ReferenceSignature { CustomerId = customer.CustomerId, ImagePath = p, FileHash = "h", CreatedAt = DateTime.UtcNow, IsActive = true });
        }
        await db.SaveChangesAsync();
        var refs = await db.ReferenceSignatures.Where(r => r.CustomerId == customer.CustomerId).ToListAsync();
        var ext = await db.ExtractedSignatures.FirstAsync(e => e.ChequeId == cheque.ChequeId);
        var vr = new VerificationResult
        {
            ChequeId = cheque.ChequeId,
            SimilarityScore = 0.5m,
            LowerThresholdUsed = 0.6585m,
            UpperThresholdUsed = 0.9150m,
            AutomaticDecision = autoDecision,
            FinalDecision = finalDecision,
            ModelName = "sig-verif-ai-v5a",
            ModelVersion = "v5a-phase7",
            VerifiedAt = DateTime.UtcNow
        };
        db.VerificationResults.Add(vr);
        await db.SaveChangesAsync();
        foreach (var r in refs)
        {
            db.SignatureComparisons.Add(new SignatureComparison
            {
                VerificationId = vr.VerificationId,
                ExtractedSignatureId = ext.ExtractedSignatureId,
                ReferenceSignatureId = r.ReferenceSignatureId,
                SimilarityScore = 0.5m,
                IsBestMatch = false
            });
        }
        await db.SaveChangesAsync();
        return vr.VerificationId;
    }

    [Fact]
    public void Controller_RequiresControleurRole()
    {
        var type = typeof(ManualReviewsController);
        var classAuth = type.GetCustomAttributes(typeof(AuthorizeAttribute), false).Cast<AuthorizeAttribute>().FirstOrDefault();
        Assert.NotNull(classAuth);
        Assert.Equal("Contrôleur", classAuth!.Roles);

        var queueMethod = type.GetMethod("Queue");
        Assert.NotNull(queueMethod);
        // Queue inherits class-level auth, no additional role
        Assert.Empty(queueMethod!.GetCustomAttributes(typeof(AuthorizeAttribute), false));

        var reviewMethod = type.GetMethod("Review");
        Assert.NotNull(reviewMethod);

        var decideMethod = type.GetMethod("Decide");
        Assert.NotNull(decideMethod);
        Assert.NotNull(decideMethod!.GetCustomAttributes(typeof(HttpPostAttribute), false).FirstOrDefault());
        Assert.NotNull(decideMethod.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), false).FirstOrDefault());
        // Ensure class-level Contrôleur is not bypassed
        var decideAuth = decideMethod.GetCustomAttributes(typeof(AuthorizeAttribute), false).Cast<AuthorizeAttribute>().FirstOrDefault();
        Assert.Null(decideAuth); // inherits from class
    }

    [Fact]
    public void Decice_RequiresAntiforgery()
    {
        var method = typeof(ManualReviewsController).GetMethod("Decide");
        Assert.NotNull(method!.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), false).FirstOrDefault());
        Assert.NotNull(method.GetCustomAttributes(typeof(HttpPostAttribute), false).FirstOrDefault());
    }

    [Fact]
    public async Task OnlyPendingManualCasesAppear()
    {
        await using var db = CreateDbContext();
        var (pendingId, _, _) = await SeedPendingManualAsync(db, 0.7625m, 3);
        await SeedNonPendingAsync(db, 1, 1, 3); // Conforme
        await SeedNonPendingAsync(db, 3, 1, 4); // already reviewed
        await SeedNonPendingAsync(db, 3, null, 3); // but cheque status 3 not 4, so not pending? Actually this one has auto 3 final null but cheque 3 -> not pending because cheque not 4
        // Create a pending but with wrong cheque status
        var cid = await SeedNonPendingAsync(db, 3, null, 3);
        // Manually fix to ensure only pendingId is valid
        var service = CreateService(db);
        var queue = await service.GetPendingQueueAsync();
        Assert.Single(queue);
        Assert.Equal(pendingId, queue.First().VerificationId);
    }

    [Fact]
    public async Task ReviewDetails_LoadsPersistedData()
    {
        await using var db = CreateDbContext();
        var (vid, _, _) = await SeedPendingManualAsync(db, 0.7625m);
        var service = CreateService(db);
        var details = await service.GetReviewDetailsAsync(vid);
        Assert.NotNull(details);
        Assert.Equal(0.7625m, details!.SimilarityScore);
        Assert.Equal(0.6585m, details.LowerThresholdUsed);
        Assert.Equal(0.9150m, details.UpperThresholdUsed);
        Assert.Equal(3, details.AutomaticDecision);
        Assert.Null(details.FinalDecision);
        Assert.Equal(4, details.ChequeStatus);
        Assert.Equal("sig-verif-ai-v5a", details.ModelName);
        Assert.Equal("v5a-phase7", details.ModelVersion);
        Assert.Equal(5, details.References.Count);
        Assert.Contains(details.References, r => r.IsBestMatch);
        Assert.NotNull(details.ExtractedSignatureImagePath);
        Assert.True(details.ChequeImageIsAccessible);
    }

    [Fact]
    public async Task Approve_ChangesFinalDecisionTo1AndChequeStatusTo3()
    {
        await using var db = CreateDbContext();
        var (vid, chequeId, _) = await SeedPendingManualAsync(db);
        var service = CreateService(db);
        var beforeVr = await db.VerificationResults.AsNoTracking().FirstAsync(v => v.VerificationId == vid);
        var beforeCheque = await db.Cheques.AsNoTracking().FirstAsync(c => c.ChequeId == chequeId);
        Assert.Null(beforeVr.FinalDecision);
        Assert.Equal((byte)4, beforeCheque.Status);

        var result = await service.DecideAsync(vid, 1, "Approuvé après contrôle visuel détaillé.", 42);

        Assert.True(result.Success);
        var afterVr = await db.VerificationResults.AsNoTracking().FirstAsync(v => v.VerificationId == vid);
        var afterCheque = await db.Cheques.AsNoTracking().FirstAsync(c => c.ChequeId == chequeId);
        Assert.Equal((byte?)1, afterVr.FinalDecision);
        Assert.Equal((byte)3, afterCheque.Status);
        Assert.Equal((byte)3, afterVr.AutomaticDecision); // unchanged
        Assert.Equal(0.7625m, afterVr.SimilarityScore);
        Assert.Equal(0.6585m, afterVr.LowerThresholdUsed);
        Assert.Equal(0.9150m, afterVr.UpperThresholdUsed);
        Assert.Equal("sig-verif-ai-v5a", afterVr.ModelName);
    }

    [Fact]
    public async Task Reject_ChangesFinalDecisionTo2AndChequeStatusTo5()
    {
        await using var db = CreateDbContext();
        var (vid, chequeId, _) = await SeedPendingManualAsync(db);
        var service = CreateService(db);
        var result = await service.DecideAsync(vid, 2, "Rejeté : signature manifestement différente.", 42);
        Assert.True(result.Success);
        var afterVr = await db.VerificationResults.AsNoTracking().FirstAsync(v => v.VerificationId == vid);
        var afterCheque = await db.Cheques.AsNoTracking().FirstAsync(c => c.ChequeId == chequeId);
        Assert.Equal((byte?)2, afterVr.FinalDecision);
        Assert.Equal((byte)5, afterCheque.Status);
    }

    [Fact]
    public async Task ReviewedByUserId_And_Comment_Stored()
    {
        await using var db = CreateDbContext();
        var (vid, _, _) = await SeedPendingManualAsync(db);
        var service = CreateService(db);
        await service.DecideAsync(vid, 1, "Commentaire test 12345", 99);
        var vr = await db.VerificationResults.AsNoTracking().FirstAsync(v => v.VerificationId == vid);
        Assert.Equal(99, vr.ReviewedByUserId);
        Assert.Equal("Commentaire test 12345", vr.ReviewerComment);
    }

    [Fact]
    public async Task AuditLog_Created()
    {
        await using var db = CreateDbContext();
        var (vid, _, _) = await SeedPendingManualAsync(db);
        var service = CreateService(db);
        await service.DecideAsync(vid, 1, "Audit test", 42);
        var audit = await db.AuditLogs.AsNoTracking().FirstOrDefaultAsync(a => a.EntityName == "VerificationResult" && a.EntityId == vid);
        Assert.NotNull(audit);
        Assert.Equal(42, audit!.UserId);
        Assert.Contains(vid.ToString(), audit.Description);
        Assert.Contains("Audit test", audit.Description);
        Assert.True(audit.Action == "MANUAL_REVIEW_APPROVE" || audit.Action == "MANUAL_REVIEW_REJECT");
    }

    [Fact]
    public async Task AutomaticDecision_RemainsUnchanged()
    {
        await using var db = CreateDbContext();
        var (vid, _, _) = await SeedPendingManualAsync(db, 0.7625m, 3);
        var service = CreateService(db);
        await service.DecideAsync(vid, 1, "commentaire valide", 1);
        var vr = await db.VerificationResults.AsNoTracking().FirstAsync(v => v.VerificationId == vid);
        Assert.Equal(3, vr.AutomaticDecision);
    }

    [Fact]
    public async Task SimilarityScore_And_Thresholds_RemainUnchanged()
    {
        await using var db = CreateDbContext();
        var (vid, _, _) = await SeedPendingManualAsync(db, 0.7625m);
        var service = CreateService(db);
        await service.DecideAsync(vid, 2, "commentaire valide pour test", 1);
        var vr = await db.VerificationResults.AsNoTracking().FirstAsync(v => v.VerificationId == vid);
        Assert.Equal(0.7625m, vr.SimilarityScore);
        Assert.Equal(0.6585m, vr.LowerThresholdUsed);
        Assert.Equal(0.9150m, vr.UpperThresholdUsed);
        Assert.Equal("sig-verif-ai-v5a", vr.ModelName);
        Assert.Equal("v5a-phase7", vr.ModelVersion);
    }

    [Fact]
    public async Task SignatureComparison_Rows_RemainUnchanged()
    {
        await using var db = CreateDbContext();
        var (vid, _, _) = await SeedPendingManualAsync(db);
        var before = await db.SignatureComparisons.AsNoTracking().Where(s => s.VerificationId == vid).OrderBy(s => s.ComparisonId).ToListAsync();
        var service = CreateService(db);
        await service.DecideAsync(vid, 1, "commentaire valide", 1);
        var after = await db.SignatureComparisons.AsNoTracking().Where(s => s.VerificationId == vid).OrderBy(s => s.ComparisonId).ToListAsync();
        Assert.Equal(before.Count, after.Count);
        for (int i = 0; i < before.Count; i++)
        {
            Assert.Equal(before[i].SimilarityScore, after[i].SimilarityScore);
            Assert.Equal(before[i].IsBestMatch, after[i].IsBestMatch);
            Assert.Equal(before[i].ReferenceSignatureId, after[i].ReferenceSignatureId);
        }
    }

    [Fact]
    public async Task AlreadyReviewed_CannotBeReviewedAgain()
    {
        await using var db = CreateDbContext();
        var (vid, _, _) = await SeedPendingManualAsync(db);
        var service = CreateService(db);
        var first = await service.DecideAsync(vid, 1, "premiere decision", 1);
        Assert.True(first.Success);
        var second = await service.DecideAsync(vid, 2, "seconde tentative", 2);
        Assert.False(second.Success);
        Assert.True(second.Message.Contains("déjà été traitée") || second.Message.Contains("n'est plus en attente"), $"Unexpected message: {second.Message}");
        var vr = await db.VerificationResults.AsNoTracking().FirstAsync(v => v.VerificationId == vid);
        Assert.Equal((byte?)1, vr.FinalDecision); // remains first decision
    }

    [Fact]
    public async Task Concurrency_OnlyOneSucceeds()
    {
        var dbName = $"concurrent-{Guid.NewGuid():N}";
        var options = new DbContextOptionsBuilder<ChequeVerificationDbContext>().UseInMemoryDatabase(dbName).Options;
        // Seed with one pending
        await using (var db = new ChequeVerificationDbContext(options))
        {
            var webRoot = Path.Combine(Path.GetTempPath(), "concurrent-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(webRoot);
            var customer = new Customer { CustomerNumber = "CUST-CONC", FullName = "Conc", AccountNumber = "ACC" };
            db.Customers.Add(customer);
            await db.SaveChangesAsync();
            var chequePath = Path.Combine(webRoot, "cheque.png");
            await File.WriteAllBytesAsync(chequePath, new byte[] { 1 });
            var cheque = new Cheque { CustomerId = customer.CustomerId, ImportedByUserId = 1, ChequeNumber = "CHQ-CONC", ImagePath = chequePath, Status = 4, UploadedAt = DateTime.UtcNow };
            db.Cheques.Add(cheque);
            await db.SaveChangesAsync();
            var extPath = Path.Combine(webRoot, "ext.png");
            await File.WriteAllBytesAsync(extPath, new byte[] { 9 });
            db.ExtractedSignatures.Add(new ExtractedSignature { ChequeId = cheque.ChequeId, ImagePath = extPath, FileHash = "h", ExtractionConfidence = 0.5m, ExtractedAt = DateTime.UtcNow });
            await db.SaveChangesAsync();
            for (int i = 0; i < 5; i++)
            {
                var p = Path.Combine(webRoot, $"ref{i}.png");
                await File.WriteAllBytesAsync(p, new byte[] { 1 });
                db.ReferenceSignatures.Add(new ReferenceSignature { CustomerId = customer.CustomerId, ImagePath = p, FileHash = $"h{i}", CreatedAt = DateTime.UtcNow, IsActive = true });
            }
            await db.SaveChangesAsync();
            var refs = await db.ReferenceSignatures.Where(r => r.CustomerId == customer.CustomerId).ToListAsync();
            var ext = await db.ExtractedSignatures.FirstAsync(e => e.ChequeId == cheque.ChequeId);
            var vr = new VerificationResult
            {
                ChequeId = cheque.ChequeId,
                SimilarityScore = 0.7m,
                LowerThresholdUsed = 0.6585m,
                UpperThresholdUsed = 0.9150m,
                AutomaticDecision = 3,
                FinalDecision = null,
                ModelName = "sig-verif-ai-v5a",
                ModelVersion = "v5a-phase7",
                VerifiedAt = DateTime.UtcNow
            };
            db.VerificationResults.Add(vr);
            await db.SaveChangesAsync();
            foreach (var r in refs)
            {
                db.SignatureComparisons.Add(new SignatureComparison { VerificationId = vr.VerificationId, ExtractedSignatureId = ext.ExtractedSignatureId, ReferenceSignatureId = r.ReferenceSignatureId, SimilarityScore = 0.5m, IsBestMatch = false });
            }
            await db.SaveChangesAsync();
            Directory.Delete(webRoot, true);
        }

        // Two concurrent services with separate contexts but same InMemory db
        var tasks = new List<Task<ManualReviewDecisionResult>>();
        for (int i = 0; i < 2; i++)
        {
            int userId = 100 + i;
            tasks.Add(Task.Run(async () =>
            {
                var opts = new DbContextOptionsBuilder<ChequeVerificationDbContext>().UseInMemoryDatabase(dbName).Options;
                await using var db2 = new ChequeVerificationDbContext(opts);
                var svc = new ManualReviewService(db2, new FakeEnv { WebRootPath = Path.GetTempPath() }, NullLogger<ManualReviewService>.Instance);
                // Need to get verificationId
                var vid = await db2.VerificationResults.AsNoTracking().Select(v => v.VerificationId).FirstAsync();
                return await svc.DecideAsync(vid, 1, $"commentaire concurrent {userId}", userId);
            }));
        }
        var results = await Task.WhenAll(tasks);
        var successCount = results.Count(r => r.Success);
        Assert.Equal(1, successCount);
        var failCount = results.Count(r => !r.Success);
        Assert.Equal(1, failCount);
        var failMsg = results.First(r => !r.Success).Message;
        Assert.True(failMsg.Contains("déjà été traitée") || failMsg.Contains("n'est plus en attente"), $"Unexpected fail message: {failMsg}");
    }

    [Fact]
    public async Task InvalidComment_Rejected()
    {
        await using var db = CreateDbContext();
        var (vid, _, _) = await SeedPendingManualAsync(db);
        var service = CreateService(db);
        var empty = await service.DecideAsync(vid, 1, "", 1);
        Assert.False(empty.Success);
        Assert.Contains("commentaire", empty.Message.ToLower());

        var shortComment = await service.DecideAsync(vid, 1, "abc", 1);
        Assert.False(shortComment.Success);

        var nullComment = await service.DecideAsync(vid, 1, "   ", 1);
        Assert.False(nullComment.Success);

        // Verify still pending
        var vr = await db.VerificationResults.AsNoTracking().FirstAsync(v => v.VerificationId == vid);
        Assert.Null(vr.FinalDecision);
    }

    [Fact]
    public async Task InvalidDecision_Rejected()
    {
        await using var db = CreateDbContext();
        var (vid, _, _) = await SeedPendingManualAsync(db);
        var service = CreateService(db);
        var result = await service.DecideAsync(vid, 3, "commentaire valide", 1); // 3 is Contrôle manuel, not allowed as final
        Assert.False(result.Success);
        Assert.Contains("Décision invalide", result.Message);
    }

    [Fact]
    public async Task Controller_Decide_RequiresAntiforgeryAndAuthorize()
    {
        var method = typeof(ManualReviewsController).GetMethod("Decide");
        Assert.NotNull(method);
        Assert.NotNull(method!.GetCustomAttributes(typeof(HttpPostAttribute), false).FirstOrDefault());
        Assert.NotNull(method.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), false).FirstOrDefault());
        var auth = method.GetCustomAttributes(typeof(AuthorizeAttribute), false).Cast<AuthorizeAttribute>().FirstOrDefault();
        // Class-level already Contrôleur, method inherits, but we check class
        var classAuth = typeof(ManualReviewsController).GetCustomAttributes(typeof(AuthorizeAttribute), false).Cast<AuthorizeAttribute>().FirstOrDefault();
        Assert.NotNull(classAuth);
        Assert.Equal("Contrôleur", classAuth!.Roles);
    }

    private sealed class FakeEnv : IWebHostEnvironment
    {
        public string ApplicationName { get; set; } = "test";
        public string EnvironmentName { get; set; } = "Development";
        public string ContentRootPath { get; set; } = ".";
        public string WebRootPath { get; set; } = ".";
        public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
        public IFileProvider WebRootFileProvider { get; set; } = new NullFileProvider();
    }
}
