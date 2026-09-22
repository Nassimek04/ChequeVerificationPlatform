using ChequeVerification.Web.Controllers;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Verifications;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

// Read-only verification history: persisted VerificationResult rows are
// listed verbatim. These tests never touch AI/OCR/OpenCV/extraction.
public class VerificationHistoryTests : IDisposable
{
    private readonly string _webRoot = Path.Combine(Path.GetTempPath(), "verify-history-" + Guid.NewGuid().ToString("N"));

    public VerificationHistoryTests() => Directory.CreateDirectory(_webRoot);
    public void Dispose() { try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { } }

    [Fact]
    public async Task History_ReturnsAllPersistedResults_NewestFirst()
    {
        await using var db = CreateDbContext();
        var oldId = await SeedResultAsync(db, "CHQ-H-OLD", verifiedAt: DateTime.UtcNow.AddDays(-2), score: 0.9500m, auto: 1, final: 1);
        var newId = await SeedResultAsync(db, "CHQ-H-NEW", verifiedAt: DateTime.UtcNow, score: 0.5000m, auto: 2, final: 2);

        var svc = CreateService(db);
        var items = await svc.GetVerificationHistoryAsync();

        Assert.Equal(2, items.Count);
        Assert.Equal(newId, items[0].VerificationId);
        Assert.Equal(oldId, items[1].VerificationId);
        Assert.True(items[0].VerifiedAt >= items[1].VerifiedAt);
    }

    [Fact]
    public async Task History_PreservesMultipleResultsForSameCheque()
    {
        await using var db = CreateDbContext();
        var customer = new Customer { CustomerNumber = "CUST-H-MULTI", FullName = "Multi", AccountNumber = "ACC" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        var cheque = new Cheque { CustomerId = customer.CustomerId, ImportedByUserId = 1, ChequeNumber = "CHQ-H-MULTI", ImagePath = "/img.png", Status = 3, UploadedAt = DateTime.UtcNow };
        db.Cheques.Add(cheque);
        await db.SaveChangesAsync();
        // Two historical rows for the same cheque (inserted directly; the
        // launch flow normally enforces one row per cheque).
        db.VerificationResults.Add(new VerificationResult
        {
            ChequeId = cheque.ChequeId, SimilarityScore = 0.7000m, LowerThresholdUsed = 0.6585m,
            UpperThresholdUsed = 0.9150m, AutomaticDecision = 3, FinalDecision = null,
            ModelName = "sig-verif-ai-v5a", ModelVersion = "v5a-phase7", VerifiedAt = DateTime.UtcNow.AddDays(-1)
        });
        db.VerificationResults.Add(new VerificationResult
        {
            ChequeId = cheque.ChequeId, SimilarityScore = 0.9200m, LowerThresholdUsed = 0.6585m,
            UpperThresholdUsed = 0.9150m, AutomaticDecision = 1, FinalDecision = 1,
            ModelName = "sig-verif-ai-v5a", ModelVersion = "v5a-phase7", VerifiedAt = DateTime.UtcNow
        });
        await db.SaveChangesAsync();

        var items = await CreateService(db).GetVerificationHistoryAsync();

        Assert.Equal(2, items.Count);
        Assert.All(items, i => Assert.Equal(cheque.ChequeId, i.ChequeId));
        Assert.Contains(items, i => i.SimilarityScore == 0.7000m);
        Assert.Contains(items, i => i.SimilarityScore == 0.9200m);
    }

    [Fact]
    public async Task History_DisplaysStoredThresholdsModelAndDecisions_Unchanged()
    {
        await using var db = CreateDbContext();
        await SeedResultAsync(db, "CHQ-H-STORED", verifiedAt: DateTime.UtcNow, score: 0.7625m,
            auto: 3, final: null, lower: 0.6000m, upper: 0.9000m,
            model: "sig-verif-ai-v5a", version: "v5a-phase7");

        var item = (await CreateService(db).GetVerificationHistoryAsync()).Single();

        // Persisted values are returned verbatim, even though they differ
        // from the current 0.6585 / 0.9150 policy.
        Assert.Equal(0.7625m, item.SimilarityScore);
        Assert.Equal(0.6000m, item.LowerThresholdUsed);
        Assert.Equal(0.9000m, item.UpperThresholdUsed);
        Assert.Equal(3, item.AutomaticDecision);
        Assert.Null(item.FinalDecision);
        Assert.Equal("sig-verif-ai-v5a", item.ModelName);
        Assert.Equal("v5a-phase7", item.ModelVersion);
        Assert.Equal("CHQ-H-STORED", item.ChequeNumber);
    }

    [Fact]
    public async Task History_ReflectsManualReviewDecisionAndReviewer()
    {
        await using var db = CreateDbContext();
        var role = new Role { Name = "Contrôleur" };
        db.Roles.Add(role);
        await db.SaveChangesAsync();
        var reviewer = new User { RoleId = role.RoleId, Email = "ctrl@test.local", FullName = "Le Contrôleur", PasswordHash = "x", Status = 1, CreatedAt = DateTime.UtcNow };
        db.Users.Add(reviewer);
        await db.SaveChangesAsync();
        await SeedResultAsync(db, "CHQ-H-REV", verifiedAt: DateTime.UtcNow, score: 0.7625m,
            auto: 3, final: 2, reviewerId: reviewer.UserId, comment: "Signature non conforme.");

        var item = (await CreateService(db).GetVerificationHistoryAsync()).Single();

        Assert.Equal((byte)3, item.AutomaticDecision);
        Assert.Equal((byte)2, item.FinalDecision);
        Assert.Equal(reviewer.UserId, item.ReviewedByUserId);
        Assert.Equal("Le Contrôleur", item.ReviewedByName);
        Assert.Equal("Signature non conforme.", item.ReviewerComment);
    }

    [Fact]
    public async Task History_DoesNotCallAiOrOcrService()
    {
        await using var db = CreateDbContext();
        await SeedResultAsync(db, "CHQ-H-NOCALL", verifiedAt: DateTime.UtcNow, score: 0.9500m, auto: 1, final: 1);

        // Throwing client: any AI/OCR call fails the test.
        var svc = new VerificationService(db, new ThrowingApiClient(),
            new FakeEnv { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance);
        var items = await svc.GetVerificationHistoryAsync();

        Assert.Single(items);
    }

    [Fact]
    public async Task History_EmptyDatabase_ReturnsEmpty()
    {
        await using var db = CreateDbContext();
        var items = await CreateService(db).GetVerificationHistoryAsync();
        Assert.Empty(items);
    }

    [Fact]
    public async Task Controller_History_IsGetOnly_AndAllowsAllThreeRoles()
    {
        var method = typeof(VerificationsController).GetMethod("History");
        Assert.NotNull(method);
        Assert.NotNull(method!.GetCustomAttributes(typeof(HttpGetAttribute), false).FirstOrDefault());
        Assert.Null(method.GetCustomAttributes(typeof(HttpPostAttribute), false).FirstOrDefault());
        var auth = method.GetCustomAttributes(typeof(AuthorizeAttribute), false).Cast<AuthorizeAttribute>().FirstOrDefault();
        Assert.NotNull(auth);
        var roles = (auth!.Roles ?? string.Empty).Split(',').Select(r => r.Trim()).ToHashSet();
        Assert.Contains("Utilisateur", roles);
        Assert.Contains("Administrateur", roles);
        Assert.Contains("Contrôleur", roles);
    }

    [Fact]
    public async Task Controller_History_ReturnsViewWithPersistedItems()
    {
        await using var db = CreateDbContext();
        await SeedResultAsync(db, "CHQ-H-CTRL", verifiedAt: DateTime.UtcNow, score: 0.9500m, auto: 1, final: 1);

        var svc = CreateService(db);
        var controller = new VerificationsController(svc, new ThrowingApiClient(), new FakeEnv { WebRootPath = _webRoot });
        var result = await controller.History();

        var view = Assert.IsType<ViewResult>(result);
        var model = Assert.IsType<VerificationHistoryViewModel>(view.Model);
        var item = Assert.Single(model.Items);
        Assert.Equal("CHQ-H-CTRL", item.ChequeNumber);
        Assert.Equal(0.9500m, item.SimilarityScore);
    }

    private async Task<int> SeedResultAsync(
        ChequeVerificationDbContext db, string chequeNumber, DateTime verifiedAt,
        decimal score, byte auto, byte? final,
        decimal lower = 0.6585m, decimal upper = 0.9150m,
        string model = "sig-verif-ai-v5a", string version = "v5a-phase7",
        int? reviewerId = null, string? comment = null)
    {
        var customer = new Customer { CustomerNumber = "CUST-" + chequeNumber, FullName = "Client " + chequeNumber, AccountNumber = "ACC" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        var cheque = new Cheque { CustomerId = customer.CustomerId, ImportedByUserId = 1, ChequeNumber = chequeNumber, ImagePath = "/img.png", Status = 3, UploadedAt = DateTime.UtcNow };
        db.Cheques.Add(cheque);
        await db.SaveChangesAsync();
        var vr = new VerificationResult
        {
            ChequeId = cheque.ChequeId, SimilarityScore = score,
            LowerThresholdUsed = lower, UpperThresholdUsed = upper,
            AutomaticDecision = auto, FinalDecision = final,
            ModelName = model, ModelVersion = version, VerifiedAt = verifiedAt,
            ReviewedByUserId = reviewerId, ReviewerComment = comment
        };
        db.VerificationResults.Add(vr);
        await db.SaveChangesAsync();
        return vr.VerificationId;
    }

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>().UseInMemoryDatabase($"history-{Guid.NewGuid():N}").Options;
    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());
    private VerificationService CreateService(ChequeVerificationDbContext db)
        => new(db, new ThrowingApiClient(), new FakeEnv { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance);

    private sealed class ThrowingApiClient : IVerificationApiClient
    {
        private static Task<T> Throw<T>() => throw new InvalidOperationException("History must not call the AI/OCR API.");
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default) => Throw<HealthResponseDto?>();
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string n, string c, CancellationToken ct = default) => Throw<ImageAnalysisResponseDto?>();
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string n, string c, CancellationToken ct = default) => Throw<SignatureExtractionResponseDto?>();
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string n, string c, CancellationToken ct = default) => Throw<SignatureDebugResponseDto?>();
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream e, string en, string ec, Stream r, string rn, string rc, CancellationToken ct = default) => Throw<SignatureComparisonResponseDto?>();
        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream s, string n, string c, CancellationToken ct = default) => Throw<ChequeOcrResponseDto?>();
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream e, string en, string ec, Stream r, string rn, string rc, CancellationToken ct = default) => Throw<SignatureAiComparisonResponseDto?>();
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
