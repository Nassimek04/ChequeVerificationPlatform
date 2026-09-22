namespace ChequeVerification.Web.Services;

/// <summary>
/// Reuses existing persisted semantics. Do NOT create duplicate enums.
/// Experimental academic thresholds — not bank validated.
/// </summary>
public static class VerificationDecision
{
    public const byte Conforme = 1;
    public const byte NonConforme = 2;
    public const byte ControleManuel = 3;
}

public static class ChequeStatus
{
    public const byte EnAttente = 1;
    public const byte EnTraitement = 2;
    public const byte Verifie = 3;
    public const byte ControleManuel = 4;
    public const byte Rejete = 5;
    public const byte Erreur = 6;
}
