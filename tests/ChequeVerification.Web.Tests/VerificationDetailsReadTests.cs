using System.Reflection;
using ChequeVerification.Web.Controllers;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

// Covers the real History -> Details read chain for Contrôleur:
//  - Details authorizes the same roles as History (no redirect to denied)
//  - #51-like persisted rows (Auto=3, Final=2, reviewer+comment) render
//    unchanged through the read-only details query (no AI/OCR/OpenCV)
//  - read-path views never offer Contrôleur an unauthorized mutation link
public class VerificationDetailsReadTests : IDisposable
{
    private readonly string _webRoot = Path.Combine(Path.GetTempPath(), "verify-details-" + Guid.NewGuid().ToString("N"));

    public VerificationDetailsReadTests() => Directory.CreateDirectory(_webRoot);
    public void Dispose() { try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { } }

    [Fact]
    public async Task Details_Returns51LikePersistedResult_Unchanged_WithoutApiCalls()
    {
        await using var db = CreateDbContext();
        var role = new Role { Name = "Contrôleur" };
        db.Roles.Add(role);
        await db.SaveChangesAsync();
        var reviewer = new User { RoleId = role.RoleId, Email = "controleur@test.local", FullName = "Controleur Test", PasswordHash = "x", Status = 1, CreatedAt = DateTime.UtcNow };
        db.Users.Add(reviewer);
        var customer = new Customer { CustomerNumber = "CUST-51", FullName = "Client 51", AccountNumber = "ACC-51" };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        var cheque = new Cheque { CustomerId = customer.CustomerId, ImportedByUserId = 1, ChequeNumber = "CHQ-0035", ImagePath = "/img.png", Status = 5, UploadedAt = DateTime.UtcNow };
        db.Cheques.Add(cheque);
        await db.SaveChangesAsync();
        var extracted = new ExtractedSignature { ChequeId = cheque.ChequeId, ImagePath = "/ext.png", FileHash = "h", ExtractionConfidence = 0.8m, ExtractedAt = DateTime.UtcNow };
        db.ExtractedSignatures.Add(extracted);
        var reference = new ReferenceSignature { CustomerId = customer.CustomerId, ImagePath = "/ref.png", FileHash = "r", CreatedAt = DateTime.UtcNow, IsActive = true };
        db.ReferenceSignatures.Add(reference);
        await db.SaveChangesAsync();
        var vr = new VerificationResult
        {
            ChequeId = cheque.ChequeId,
            SimilarityScore = 0.7625m,
            LowerThresholdUsed = 0.6585m,
            UpperThresholdUsed = 0.9150m,
            AutomaticDecision = 3,
            FinalDecision = 2,
            ModelName = "sig-verif-ai-v5a",
            ModelVersion = "v5a-phase7",
            VerifiedAt = DateTime.UtcNow,
            ReviewedByUserId = reviewer.UserId,
            ReviewerComment = "Signature non conforme."
        };
        db.VerificationResults.Add(vr);
        await db.SaveChangesAsync();
        db.SignatureComparisons.Add(new SignatureComparison
        {
            VerificationId = vr.VerificationId,
            ExtractedSignatureId = extracted.ExtractedSignatureId,
            ReferenceSignatureId = reference.ReferenceSignatureId,
            SimilarityScore = 0.7625m,
            IsBestMatch = true
        });
        await db.SaveChangesAsync();

        // Throwing client proves no AI/OCR/OpenCV call happens on read.
        var svc = new VerificationService(db, new ThrowingApiClient(),
            new FakeEnv { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance);
        var details = await svc.GetVerificationDetailsAsync(vr.VerificationId);

        Assert.NotNull(details);
        Assert.Equal(0.7625m, details!.SimilarityScore);
        Assert.Equal(0.6585m, details.LowerThresholdUsed);
        Assert.Equal(0.9150m, details.UpperThresholdUsed);
        Assert.Equal((byte)3, details.AutomaticDecision);
        Assert.Equal((byte)2, details.FinalDecision);
        Assert.Equal("sig-verif-ai-v5a", details.ModelName);
        Assert.Equal("v5a-phase7", details.ModelVersion);
        Assert.Equal("Controleur Test", details.ReviewedByName);
        Assert.Equal("Signature non conforme.", details.ReviewerComment);
        Assert.Equal("CHQ-0035", details.ChequeNumber);
        var comparison = Assert.Single(details.Comparisons);
        Assert.Equal(0.7625m, comparison.SimilarityScore);
        Assert.True(comparison.IsBestMatch);
        // Persisted row untouched by the read.
        Assert.Equal((byte)2, (await db.VerificationResults.SingleAsync()).FinalDecision);
    }

    [Fact]
    public void Details_EffectivePolicy_MatchesHistory()
    {
        // Same stacked policy object content on both actions: whatever passes
        // History necessarily passes Details (no redirect to denied between
        // the two links of the chain).
        Assert.Equal(
            EffectiveRoles(typeof(VerificationsController), "History"),
            EffectiveRoles(typeof(VerificationsController), "Details"));
        Assert.Contains("Contrôleur", EffectiveRoles(typeof(VerificationsController), "Details"));
    }

    [Fact]
    public void HistoryView_DoesNotOfferCreate_ToControleur()
    {
        var view = ReadView("Views/Verifications/History.cshtml");
        // The Create button must sit inside a Utilisateur/Administrateur gate
        // (same convention as Cheques/Index), never unconditional.
        var gateIndex = view.IndexOf("Nouvelle vérification", StringComparison.Ordinal);
        Assert.True(gateIndex > 0);
        var head = view[..gateIndex];
        Assert.Contains("IsInRole(\"Utilisateur\")", head);
        Assert.Contains("IsInRole(\"Administrateur\")", head);
    }

    [Fact]
    public void DetailsView_ControleurRetour_GoesToHistory_NotCreate()
    {
        var view = ReadView("Views/Verifications/Details.cshtml");
        // A Contrôleur gate must exist and the read-only History target must
        // be offered after it; Utilisateur/Administrateur keep Retour->Create.
        var gateIndex = view.IndexOf("IsInRole(\"Contrôleur\")", StringComparison.Ordinal);
        Assert.True(gateIndex >= 0);
        Assert.Contains("asp-action=\"History\"", view[gateIndex..]);
    }

    private static HashSet<string> EffectiveRoles(Type controller, string actionName)
    {
        var method = controller.GetMethod(actionName)
            ?? throw new InvalidOperationException($"Action {controller.Name}.{actionName} not found.");
        HashSet<string>? effective = null;
        foreach (var attr in controller.GetCustomAttributes<AuthorizeAttribute>(false)
                     .Concat(method.GetCustomAttributes<AuthorizeAttribute>(false)))
        {
            if (string.IsNullOrWhiteSpace(attr.Roles))
                continue;
            var roles = attr.Roles.Split(',').Select(r => r.Trim())
                .Where(r => r.Length > 0).ToHashSet();
            effective = effective == null ? roles : new HashSet<string>(effective.Intersect(roles));
        }
        return effective ?? new HashSet<string>();
    }

    private static string ReadView(string relativePath)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "ChequeVerificationPlatform.slnx")))
            dir = dir.Parent;
        Assert.True(dir != null, "Solution root not found.");
        var path = Path.Combine(dir!.FullName, "src/ChequeVerification.Web", relativePath.Replace('/', Path.DirectorySeparatorChar));
        Assert.True(File.Exists(path), $"View not found: {path}");
        return File.ReadAllText(path);
    }

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>().UseInMemoryDatabase($"details-{Guid.NewGuid():N}").Options;
    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());

    private sealed class ThrowingApiClient : IVerificationApiClient
    {
        private static Task<T> Throw<T>() => throw new InvalidOperationException("Details/History must not call the AI/OCR API.");
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default) => Throw<HealthResponseDto?>();
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string n, string c, CancellationToken ct = default) => Throw<ImageAnalysisResponseDto?>();
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string n, string c, CancellationToken ct = default) => Throw<SignatureExtractionResponseDto?>();
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string n, string c, CancellationToken ct = default) => Throw<SignatureDebugResponseDto?>();
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream e, string en, string ec, Stream r, string rn, string rc, CancellationToken ct = default) => Throw<SignatureComparisonResponseDto?>();
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream e, string en, string ec, Stream r, string rn, string rc, CancellationToken ct = default) => Throw<SignatureAiComparisonResponseDto?>();
        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream s, string n, string c, CancellationToken ct = default) => Throw<ChequeOcrResponseDto?>();
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
