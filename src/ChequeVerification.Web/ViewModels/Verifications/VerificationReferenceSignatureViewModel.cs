namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationReferenceSignatureViewModel
{
    public int ReferenceSignatureId { get; set; }
    public string ImagePath { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public bool IsActive { get; set; }
    public bool ImageIsAccessible { get; set; }
}