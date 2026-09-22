using ChequeVerification.Web.ViewModels.Verifications;

namespace ChequeVerification.Web.Services;

public class SignatureComparisonOperationResult
{
    public bool Success { get; set; }

    public string Message { get; set; } = string.Empty;

    public List<VerificationReferenceComparisonViewModel> Comparisons { get; set; } = new();

    public int? BestReferenceSignatureId { get; set; }

    public double? BestSimilarityScore { get; set; }

    public string Method { get; set; } = string.Empty;

    public string Version { get; set; } = string.Empty;
}
