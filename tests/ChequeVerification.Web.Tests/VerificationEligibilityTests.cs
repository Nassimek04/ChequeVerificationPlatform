using ChequeVerification.Web.Data;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

public class VerificationEligibilityTests
{
    [Fact]
    public async Task GetAvailableCheques_PendingChequeAppears()
    {
        await using var db = CreateDbContext();
        await SeedChequeAsync(db, ChequeStatus.EnAttente);
        var service = CreateService(db, new RejectingApiClient());

        var available = await service.GetAvailableChequesAsync();

        var cheque = Assert.Single(available);
        Assert.Equal(ChequeStatus.EnAttente, cheque.Status);
    }

    [Theory]
    [InlineData(ChequeStatus.EnTraitement)]
    [InlineData(ChequeStatus.Verifie)]
    [InlineData(ChequeStatus.ControleManuel)]
    [InlineData(ChequeStatus.Rejete)]
    [InlineData(ChequeStatus.Erreur)]
    public async Task GetAvailableCheques_NonPendingChequeDoesNotAppear(byte status)
    {
        await using var db = CreateDbContext();
        await SeedChequeAsync(db, status);
        var service = CreateService(db, new RejectingApiClient());

        var available = await service.GetAvailableChequesAsync();

        Assert.Empty(available);
    }

    [Fact]
    public async Task DirectAttempt_AgainstNonEligibleChequeIsRejectedBeforeApiCalls()
    {
        await using var db = CreateDbContext();
        var chequeId = await SeedChequeAsync(db, ChequeStatus.Erreur);
        var api = new RejectingApiClient();
        var service = CreateService(db, api);

        var preparation = await service.PrepareVerificationAsync(chequeId);
        var extraction = await service.ExtractAndPersistSignatureAsync(chequeId, 1);
        var launch = await service.LaunchVerificationAsync(chequeId, 1);

        Assert.NotNull(preparation);
        Assert.False(preparation!.CanStartVerification);
        Assert.Contains("pas en attente", preparation.BlockingReason, StringComparison.OrdinalIgnoreCase);
        Assert.False(extraction.Success);
        Assert.False(launch.Success);
        Assert.Equal(0, api.ExtractionCallCount);
        Assert.Equal(0, api.AiComparisonCallCount);
        Assert.False(await db.VerificationResults.AnyAsync());
    }

    private static ChequeVerificationDbContext CreateDbContext()
    {
        var options = new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"verification-eligibility-{Guid.NewGuid():N}")
            .Options;
        return new ChequeVerificationDbContext(options);
    }

    private static VerificationService CreateService(ChequeVerificationDbContext db, IVerificationApiClient api)
        => new(db, api, new FakeWebHostEnvironment(), NullLogger<VerificationService>.Instance);

    private static async Task<int> SeedChequeAsync(ChequeVerificationDbContext db, byte status)
    {
        var customer = new Customer
        {
            CustomerNumber = $"CUST-{status}",
            FullName = "Client éligibilité",
            AccountNumber = $"ACC-{status}"
        };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();

        var cheque = new Cheque
        {
            CustomerId = customer.CustomerId,
            ImportedByUserId = 1,
            ChequeNumber = $"CHQ-STATUS-{status}",
            ImagePath = "/uploads/cheques/eligibility.png",
            Status = status,
            UploadedAt = DateTime.UtcNow
        };
        db.Cheques.Add(cheque);
        await db.SaveChangesAsync();
        return cheque.ChequeId;
    }

    private sealed class RejectingApiClient : IVerificationApiClient
    {
        public int ExtractionCallCount { get; private set; }
        public int AiComparisonCallCount { get; private set; }

        public Task<HealthResponseDto?> GetHealthAsync(CancellationToken cancellationToken = default)
            => Task.FromResult<HealthResponseDto?>(null);

        public Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<ImageAnalysisResponseDto?>(null);

        public Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
        {
            ExtractionCallCount++;
            return Task.FromResult<SignatureExtractionResponseDto?>(null);
        }

        public Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<SignatureDebugResponseDto?>(null);

        public Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default)
            => Task.FromResult<SignatureComparisonResponseDto?>(null);

        public Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default)
        {
            AiComparisonCallCount++;
            return Task.FromResult<SignatureAiComparisonResponseDto?>(null);
        }

        public Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
            => Task.FromResult<ChequeOcrResponseDto?>(null);
    }

    private sealed class FakeWebHostEnvironment : IWebHostEnvironment
    {
        public string ApplicationName { get; set; } = "ChequeVerification.Web.Tests";
        public string EnvironmentName { get; set; } = "Development";
        public string ContentRootPath { get; set; } = ".";
        public string WebRootPath { get; set; } = ".";
        public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
        public IFileProvider WebRootFileProvider { get; set; } = new NullFileProvider();
    }
}
