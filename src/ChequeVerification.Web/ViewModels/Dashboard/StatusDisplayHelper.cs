namespace ChequeVerification.Web.ViewModels.Dashboard;

public static class StatusDisplayHelper
{
    public static string ChequeStatusLabel(byte status) => status switch
    {
        1 => "En attente",
        2 => "En traitement",
        3 => "Vérifié",
        4 => "Contrôle manuel",
        5 => "Rejeté",
        6 => "Erreur",
        _ => "Inconnu"
    };

    public static string ChequeStatusBadge(byte status) => status switch
    {
        1 => "bg-secondary",
        2 => "bg-info text-dark",
        3 => "bg-success",
        4 => "bg-warning text-dark",
        5 => "bg-danger",
        6 => "bg-dark",
        _ => "bg-secondary"
    };

    public static string VerificationDecisionLabel(byte? decision) => decision switch
    {
        1 => "Conforme",
        2 => "Non conforme",
        3 => "Contrôle manuel",
        null => "En attente",
        _ => "Inconnu"
    };

    public static string VerificationDecisionBadge(byte? decision) => decision switch
    {
        1 => "bg-success",
        2 => "bg-danger",
        3 => "bg-warning text-dark",
        null => "bg-secondary",
        _ => "bg-secondary"
    };
}