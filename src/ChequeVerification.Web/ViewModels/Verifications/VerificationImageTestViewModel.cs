namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationImageTestViewModel
{
    public int ChequeId { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public int? Width { get; set; }
    public int? Height { get; set; }
    public int? Channels { get; set; }
    public string? ContentType { get; set; }
    public bool Decoded { get; set; }
    public bool GrayscaleReady { get; set; }
}