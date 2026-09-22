namespace ChequeVerification.Web.ViewModels.ReferenceSignatures;

public class ReferenceSignatureListViewModel
{
    public string? SearchTerm { get; set; }
    public List<ReferenceSignatureListItemViewModel> Signatures { get; set; } = new();
    public int TotalCount { get; set; }
    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 10;
    public int TotalPages => PageSize <= 0 ? 0 : (int)Math.Ceiling(TotalCount / (double)PageSize);
}