namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationSignatureComparisonViewModel
{
    public int ChequeId { get; set; }
    public bool Success { get; set; }
    public string? Message { get; set; }
    public List<VerificationReferenceComparisonViewModel> Comparisons { get; set; } = new();
    public int? BestReferenceSignatureId { get; set; }
    public double? BestSimilarityScore { get; set; }
    public string Method { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
}

public class VerificationReferenceComparisonViewModel
{
    public int ReferenceSignatureId { get; set; }
    public bool IsAvailable { get; set; }
    public string? StatusMessage { get; set; }
    public double? Score { get; set; }
    public string Method { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
}
