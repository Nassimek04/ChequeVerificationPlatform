using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace ChequeVerification.Web.Tests;

public class VerificationPolicyTests : IDisposable
{
    private readonly string _webRoot = Path.Combine(Path.GetTempPath(), "verify-policy-" + Guid.NewGuid().ToString("N"));

    public VerificationPolicyTests() => Directory.CreateDirectory(_webRoot);
    public void Dispose() { try { if (Directory.Exists(_webRoot)) Directory.Delete(_webRoot, true); } catch { } }

    [Theory]
    [InlineData(0.6584, 2)] // Non conforme
    [InlineData(0.6585, 2)] // boundary inclusive L
    [InlineData(0.6586, 3)] // Controle manuel
    [InlineData(0.9149, 3)]
    [InlineData(0.9150, 1)] // Conforme inclusive U
    [InlineData(0.9151, 1)]
    public async Task Policy_Boundaries(double score, byte expectedDecision)
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, 5);
        var api = new FakeAiClient(_ => Success(score));
        var policy = Options.Create(new VerificationPolicyOptions { LowerThreshold = 0.6585m, UpperThreshold = 0.9150m });
        var svc = CreateService(db, api, policy);
        var res = await svc.LaunchVerificationAsync(1, 1);
        Assert.True(res.Success);
        Assert.Equal(expectedDecision, res.AutomaticDecision);
        var vr = await db.VerificationResults.SingleAsync(v => v.ChequeId == 1);
        Assert.Equal(expectedDecision, vr.AutomaticDecision);
    }

    [Fact]
    public async Task Policy_Invalid_L_GreaterEqual_U_Fails()
    {
        await using var db = CreateDbContext();
        await SeedAsync(db, 1);
        var api = new FakeAiClient(_ => Success(0.5));
        var badPolicy = Options.Create(new VerificationPolicyOptions { LowerThreshold = 0.8m, UpperThreshold = 0.7m });
        var svc = CreateService(db, api, badPolicy);
        var res = await svc.LaunchVerificationAsync(1, 1);
        Assert.False(res.Success);
        Assert.Contains("Configuration", res.Message);
        Assert.False(await db.VerificationResults.AnyAsync());
    }

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>().UseInMemoryDatabase($"policy-{Guid.NewGuid():N}").Options;
    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());
    private VerificationService CreateService(ChequeVerificationDbContext db, IVerificationApiClient api, IOptions<VerificationPolicyOptions> policy)
        => new(db, api, new FakeEnv { WebRootPath = _webRoot }, NullLogger<VerificationService>.Instance, policy);

    private async Task SeedAsync(ChequeVerificationDbContext db, int refs)
    {
        var cust = new Customer { CustomerNumber = "CUST-POL", FullName = "Pol", AccountNumber = "ACC" };
        db.Customers.Add(cust); await db.SaveChangesAsync();
        for (int i = 1; i <= refs; i++)
        {
            var p = Path.Combine(_webRoot, $"ref{i}.png"); await File.WriteAllBytesAsync(p, new byte[] { 1, 2, 3 });
            db.ReferenceSignatures.Add(new ReferenceSignature { ReferenceSignatureId = i, CustomerId = cust.CustomerId, ImagePath = p, CreatedAt = DateTime.UtcNow.AddMinutes(-i), IsActive = true });
        }
        var chequePath = Path.Combine(_webRoot, "cheque.png"); await File.WriteAllBytesAsync(chequePath, new byte[] { 1 });
        db.Cheques.Add(new Cheque { ChequeId = 1, CustomerId = cust.CustomerId, ImportedByUserId = 1, ChequeNumber = "CHQ-POL", ImagePath = chequePath, Status = 1, UploadedAt = DateTime.UtcNow });
        var extPath = Path.Combine(_webRoot, "ext.png"); await File.WriteAllBytesAsync(extPath, new byte[] { 9 });
        db.ExtractedSignatures.Add(new ExtractedSignature { ChequeId = 1, ImagePath = extPath, FileHash = "h", ExtractionConfidence = 0.5m, ExtractedAt = DateTime.UtcNow });
        await db.SaveChangesAsync();
    }
    private static SignatureAiComparisonResponseDto Success(double s) => new() { Success = true, SimilarityScore = s, Method = "ai_metric", Version = "v2", Model = "siamese_resnet18", Device = "cpu", Message = "ok" };

    private sealed class FakeAiClient : ChequeVerification.Web.Services.Interfaces.IVerificationApiClient
    {
        private readonly Func<int, SignatureAiComparisonResponseDto?> _fn;
        public FakeAiClient(Func<int, SignatureAiComparisonResponseDto?> fn) => _fn = fn;
        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken ct = default) => Task.FromResult<HealthResponseDto?>(new HealthResponseDto { Status = "ok" });
        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<ImageAnalysisResponseDto?>(null);
        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureExtractionResponseDto?>(null);
        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<SignatureDebugResponseDto?>(null);
        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream a, string b, string c, Stream d, string e, string f, CancellationToken ct = default) => Task.FromResult<SignatureComparisonResponseDto?>(null);
        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream es, string ef, string ec, Stream rs, string rf, string rc, CancellationToken ct = default)
        {
            var name = Path.GetFileNameWithoutExtension(rf);
            int id = int.TryParse(name.Replace("ref", ""), out var p) ? p : 1;
            return Task.FromResult(_fn(id));
        }
        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream s, string f, string c, CancellationToken ct = default) => Task.FromResult<ChequeOcrResponseDto?>(null);
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
