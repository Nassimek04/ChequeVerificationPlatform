namespace ChequeVerification.Web.ViewModels.Shared;

/// <summary>
/// Presentation-only French labels for persisted audit action codes.
/// The database values are never modified; unknown codes fall back
/// safely to the original stored value.
/// </summary>
public static class AuditActionLabels
{
    private static readonly Dictionary<string, string> Labels = new(StringComparer.Ordinal)
    {
        ["Login"] = "Connexion",
        ["Logout"] = "Déconnexion",
        ["IMPORT_CHEQUE"] = "Import du chèque",
        ["EXTRACT_SIGNATURE"] = "Extraction de signature",
        ["REEXTRACT_SIGNATURE"] = "Réextraction de signature",
        ["VERIFY_CHEQUE"] = "Vérification du chèque",
        ["ENROLL_REFERENCE"] = "Enrôlement d'une signature de référence",
        ["MANUAL_REVIEW_APPROVE"] = "Validation manuelle",
        ["MANUAL_REVIEW_REJECT"] = "Rejet manuel"
    };

    public static string Label(string? action)
    {
        if (string.IsNullOrWhiteSpace(action))
        {
            return "—";
        }

        return Labels.TryGetValue(action, out var label) ? label : action;
    }
}
