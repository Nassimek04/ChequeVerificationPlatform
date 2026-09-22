using ChequeVerification.Web.ViewModels.Verifications;

namespace ChequeVerification.Web.Services;

public class ChequeOcrOperationResult
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public string FullText { get; set; } = string.Empty;
    public List<VerificationOcrLineViewModel> Lines { get; set; } = new();
    public VerificationOcrFieldsViewModel Fields { get; set; } = new();
    public int ProcessingMs { get; set; }
    public string Lang { get; set; } = string.Empty;
    public string Device { get; set; } = string.Empty;
}
