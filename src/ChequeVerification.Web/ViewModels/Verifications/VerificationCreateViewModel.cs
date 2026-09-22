namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationCreateViewModel
{
    public int? SelectedChequeId { get; set; }
    public List<VerificationChequeOptionViewModel> AvailableCheques { get; set; } = new();
    public VerificationPreparationViewModel? Preparation { get; set; }
    public bool AnalysisServiceAvailable { get; set; }
    public VerificationImageTestViewModel? ImageTest { get; set; }
    public VerificationSignatureTestViewModel? SignatureTest { get; set; }
    public VerificationSignatureDebugViewModel? SignatureDebug { get; set; }
    public VerificationSignatureExtractionViewModel? SignatureExtraction { get; set; }
    public VerificationSignatureComparisonViewModel? SignatureComparison { get; set; }
    public VerificationSignatureAiComparisonViewModel? SignatureAiComparison { get; set; }
    public VerificationChequeOcrViewModel? ChequeOcr { get; set; }
}