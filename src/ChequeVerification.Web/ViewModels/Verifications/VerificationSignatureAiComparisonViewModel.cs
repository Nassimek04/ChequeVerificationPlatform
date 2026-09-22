namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationSignatureAiComparisonViewModel
{
    public int ChequeId { get; set; }
    public bool Success { get; set; }
    public string? Message { get; set; }
    public List<VerificationReferenceAiComparisonViewModel> Comparisons { get; set; } = new();

    public double? MeanRawScore { get; set; }

    public int ActiveReferenceCount { get; set; }

    public int ComparedReferenceCount { get; set; }

    public int UnavailableReferenceCount { get; set; }

    public bool IsAggregationAvailable { get; set; }
}

public class VerificationReferenceAiComparisonViewModel
{
    public int ReferenceSignatureId { get; set; }
    public bool IsAvailable { get; set; }
    public string? StatusMessage { get; set; }
    public double? Score { get; set; }
    public string Method { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string Model { get; set; } = string.Empty;
    public string Device { get; set; } = string.Empty;
}
