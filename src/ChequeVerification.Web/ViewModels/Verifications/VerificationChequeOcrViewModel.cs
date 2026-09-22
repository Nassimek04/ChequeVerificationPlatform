namespace ChequeVerification.Web.ViewModels.Verifications;

public class VerificationChequeOcrViewModel
{
    public int ChequeId { get; set; }
    public bool Success { get; set; }
    public string? Message { get; set; }
    public string FullText { get; set; } = string.Empty;
    public List<VerificationOcrLineViewModel> Lines { get; set; } = new();
    public VerificationOcrFieldsViewModel Fields { get; set; } = new();
    public int ProcessingMs { get; set; }
    public string Lang { get; set; } = string.Empty;
    public string Device { get; set; } = string.Empty;
}

public class VerificationOcrLineViewModel
{
    public string Text { get; set; } = string.Empty;
    public double Confidence { get; set; }
    public List<List<double>> Box { get; set; } = new();
}

public class VerificationOcrFieldsViewModel
{
    public string? ChequeNumber { get; set; }
    public string? Date { get; set; }
    public string? AmountText { get; set; }
    public string? AmountNumeric { get; set; }
    public string? AccountNumber { get; set; }
}
