namespace ChequeVerification.Web.ViewModels.ReferenceSignatures;

public class ReferenceSignatureDetailsViewModel
{
    public int ReferenceSignatureId { get; set; }
    public int CustomerId { get; set; }
    public string CustomerNumber { get; set; } = string.Empty;
    public string CustomerFullName { get; set; } = string.Empty;
    public string ImagePath { get; set; } = string.Empty;
    public string? FileHash { get; set; }
    public bool IsActive { get; set; }
    public DateTime CreatedAt { get; set; }
    public bool ImageIsAccessible { get; set; }
}