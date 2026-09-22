namespace ChequeVerification.Web.ViewModels.Cheques;

public class ChequeListItemViewModel
{
    public int ChequeId { get; set; }
    public string ChequeNumber { get; set; } = string.Empty;
    public string CustomerNumber { get; set; } = string.Empty;
    public string CustomerFullName { get; set; } = string.Empty;
    public string AccountNumber { get; set; } = string.Empty;
    public byte Status { get; set; }
    public DateTime UploadedAt { get; set; }
    public string ImportedByFullName { get; set; } = string.Empty;
}