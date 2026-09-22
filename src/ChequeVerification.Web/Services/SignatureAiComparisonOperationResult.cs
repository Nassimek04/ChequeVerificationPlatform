using ChequeVerification.Web.ViewModels.Verifications;

namespace ChequeVerification.Web.Services;

public class SignatureAiComparisonOperationResult
{
    public bool Success { get; set; }

    public string Message { get; set; } = string.Empty;

    public List<VerificationReferenceAiComparisonViewModel> Comparisons { get; set; } = new();

    public double? MeanRawScore { get; set; }

    public int ActiveReferenceCount { get; set; }

    public int ComparedReferenceCount { get; set; }

    public int UnavailableReferenceCount { get; set; }

    public bool IsAggregationAvailable { get; set; }
}
