namespace ChequeVerification.Web.Services;

/// <summary>
/// Experimental academic thresholds — not bank validated.
/// Configured via appsettings.json VerificationPolicy section.
/// </summary>
public class VerificationPolicyOptions
{
    public const string SectionName = "VerificationPolicy";

    /// <summary>Lower threshold L = 0.6585 (rounded decimal(5,4) of 0.6585003733634949) — experimental prototype</summary>
    public decimal LowerThreshold { get; set; } = 0.6585m;

    /// <summary>Upper threshold U = 0.9150 (rounded decimal(5,4) of 0.9150440096855164) — experimental prototype</summary>
    public decimal UpperThreshold { get; set; } = 0.9150m;

    /// <summary>Canonical full-precision values for documentation: L 0.6585003733634949 U 0.9150440096855164. Runtime uses decimal(5,4) rounded values to avoid boundary mismatch between config and DB persistence.</summary>

    public string ModelName { get; set; } = "sig-verif-ai-v5a";

    public string ModelVersion { get; set; } = "v5a-phase7";

    public bool IsValid(out string? error)
    {
        if (LowerThreshold < 0m || LowerThreshold > 1m)
        {
            error = "LowerThreshold must be in [0,1].";
            return false;
        }
        if (UpperThreshold < 0m || UpperThreshold > 1m)
        {
            error = "UpperThreshold must be in [0,1].";
            return false;
        }
        if (LowerThreshold >= UpperThreshold)
        {
            error = "LowerThreshold must be < UpperThreshold (0 <= L < U <= 1).";
            return false;
        }
        error = null;
        return true;
    }
}
