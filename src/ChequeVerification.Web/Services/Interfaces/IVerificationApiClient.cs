using ChequeVerification.Web.Dtos.VerificationApi;

namespace ChequeVerification.Web.Services.Interfaces;

public interface IVerificationApiClient
{
    Task<HealthResponseDto?> GetHealthAsync(CancellationToken cancellationToken = default);
    Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default);
    Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default);
    Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default);
    Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default);
    Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default);
    Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default);
}