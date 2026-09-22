using ChequeVerification.Web.Services;
using ChequeVerification.Web.ViewModels.Verifications;

namespace ChequeVerification.Web.Services.Interfaces;

public interface IVerificationService
{
    Task<IReadOnlyList<VerificationChequeOptionViewModel>> GetAvailableChequesAsync(CancellationToken cancellationToken = default);
    Task<VerificationPreparationViewModel?> PrepareVerificationAsync(int chequeId, CancellationToken cancellationToken = default);
    Task<string?> GetChequeImagePathAsync(int chequeId, CancellationToken cancellationToken = default);

    Task<SignatureExtractionOperationResult> ExtractAndPersistSignatureAsync(
        int chequeId,
        int? userId,
        CancellationToken cancellationToken = default);

    Task<SignatureExtractionOperationResult> ReExtractAndPersistSignatureAsync(
        int chequeId,
        int? userId,
        CancellationToken cancellationToken = default);

    Task<SignatureComparisonOperationResult> CompareSignaturesWithReferencesAsync(
        int chequeId,
        CancellationToken cancellationToken = default);

    Task<SignatureAiComparisonOperationResult> CompareAiSignaturesWithReferencesAsync(
        int chequeId,
        CancellationToken cancellationToken = default);

    Task<ChequeOcrOperationResult> OcrChequeAsync(
        int chequeId,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Best-effort live V2.4 extraction diagnostic for Bilan display only.
    /// Read-only: no DB write, no Status change, no audit, no file write.
    /// Never historical evidence and never an input to the decision.
    /// </summary>
    Task<VerificationSignatureDebugViewModel> GetExtractionDiagnosticAsync(
        int chequeId,
        CancellationToken cancellationToken = default);

    Task<LaunchVerificationOperationResult> LaunchVerificationAsync(
        int chequeId,
        int currentUserId,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Automatic User flow: Status=1 only. Reuses ExtractAndPersistSignatureAsync
    /// (OpenCV V2.4) then LaunchVerificationAsync (V5-A K=5, L/U policy).
    /// No algorithm rebuild. OCR stays diagnostic and is rendered live in Bilan.
    /// </summary>
    Task<LaunchVerificationOperationResult> VerifyAutomaticallyAsync(
        int chequeId,
        int currentUserId,
        CancellationToken cancellationToken = default);

    Task<VerificationDetailsViewModel?> GetVerificationDetailsAsync(
        int verificationId,
        CancellationToken cancellationToken = default);

    Task<VerificationDetailsViewModel?> GetVerificationDetailsByChequeAsync(
        int chequeId,
        CancellationToken cancellationToken = default);

    // Read-only history: returns ALL persisted VerificationResults
    // (newest first). Never reruns AI/OCR/OpenCV/extraction.
    Task<IReadOnlyList<VerificationHistoryItemViewModel>> GetVerificationHistoryAsync(
        CancellationToken cancellationToken = default);
}