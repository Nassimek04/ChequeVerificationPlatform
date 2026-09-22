using System.Globalization;

namespace ChequeVerification.Web.ViewModels.Dashboard;

/// <summary>
/// Presentation-only Moroccan Dirham formatting. The platform operates with
/// Moroccan cheques; persisted amounts already represent dirhams, so this
/// helper formats values without any conversion: "12 500,00 MAD".
/// Isolated fr-FR numeric formatting — application-wide culture is untouched.
/// </summary>
public static class CurrencyDisplayHelper
{
    private static readonly CultureInfo Fr = CultureInfo.GetCultureInfo("fr-FR");

    public static string FormatMad(decimal amount) => amount.ToString("N2", Fr) + " MAD";

    public static string FormatMad(decimal? amount) => amount.HasValue ? FormatMad(amount.Value) : "—";
}
