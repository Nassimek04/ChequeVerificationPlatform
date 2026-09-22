using ChequeVerification.Web.ViewModels.Cheques;

namespace ChequeVerification.Web.Services.Interfaces;

/// <summary>
/// Phase 1 batch cheque import. OCR is reused via <see cref="IVerificationApiClient"/>
/// (existing PaddleOCR pipeline); this service only derives import metadata
/// (cheque number, account number, amount) and maps accounts to existing customers.
/// No automatic verification is performed here.
/// </summary>
public interface IChequeBatchImportService
{
    /// <summary>
    /// Validates uploaded files, stages them to a temp folder, runs OCR per image
    /// and builds preview rows. Nothing is inserted.
    /// </summary>
    Task<BatchImportPreviewViewModel> BuildPreviewAsync(
        IList<IFormFile> files, string webRootPath, CancellationToken cancellationToken = default);

    /// <summary>
    /// Inserts valid rows with partial-success semantics (one bad row never
    /// cancels the batch). Returns per-row outcomes.
    /// </summary>
    Task<BatchImportResultViewModel> ConfirmImportAsync(
        BatchImportConfirmViewModel model, int userId, string webRootPath, CancellationToken cancellationToken = default);
}
