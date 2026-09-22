namespace ChequeVerification.Web.ViewModels.ReferenceSignatures;

public class ReferenceSignatureListItemViewModel
{
    public int ReferenceSignatureId { get; set; }
    public string CustomerNumber { get; set; } = string.Empty;
    public string CustomerFullName { get; set; } = string.Empty;
    public string ImagePath { get; set; } = string.Empty;
    public bool IsActive { get; set; }
    public DateTime CreatedAt { get; set; }
}