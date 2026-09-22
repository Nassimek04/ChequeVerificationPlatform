namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationSignatureTestViewModel
{
    public int ChequeId { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public int? OriginalWidth { get; set; }
    public int? OriginalHeight { get; set; }
    public int? RoiX { get; set; }
    public int? RoiY { get; set; }
    public int? RoiWidth { get; set; }
    public int? RoiHeight { get; set; }
    public int? BboxX { get; set; }
    public int? BboxY { get; set; }
    public int? BboxWidth { get; set; }
    public int? BboxHeight { get; set; }
    public double? ExtractionQuality { get; set; }
    public string SignatureImageDataUri { get; set; } = string.Empty;
}