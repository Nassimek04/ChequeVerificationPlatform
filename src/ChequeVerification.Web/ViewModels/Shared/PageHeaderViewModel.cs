namespace ChequeVerification.Web.ViewModels.Shared;

/// <summary>
/// Presentation-only model for the shared page-header partial.
/// </summary>
public class PageHeaderViewModel
{
    public string Title { get; set; } = string.Empty;
    public string? Eyebrow { get; set; }
    public string? Subtitle { get; set; }
    public string? ActionsHtml { get; set; }
}
