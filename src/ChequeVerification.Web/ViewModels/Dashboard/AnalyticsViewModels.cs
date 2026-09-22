namespace ChequeVerification.Web.ViewModels.Dashboard;

/// <summary>
/// Read-only analytics DTOs aggregated from persisted records only.
/// No business-rule, persistence, or workflow change.
/// </summary>
public class VerificationActivitySeries
{
    public List<string> Labels { get; set; } = new();
    public List<int> Counts { get; set; } = new();
    public int WindowDays { get; set; } = 14;
    public int Total => Counts.Sum();
}

public class DecisionDistribution
{
    public int Conforme { get; set; }
    public int ControleManuel { get; set; }
    public int NonConforme { get; set; }
    public int Total => Conforme + ControleManuel + NonConforme;
}

/// <summary>
/// Descriptive grouping of persisted AI scores against each record's OWN
/// persisted thresholds (LowerThresholdUsed / UpperThresholdUsed).
/// Historical records are never reinterpreted with today's policy values.
/// </summary>
public class ScoreZoneDistribution
{
    public int ZoneNonConforme { get; set; }
    public int ZoneManuelle { get; set; }
    public int ZoneConforme { get; set; }
    public int Total => ZoneNonConforme + ZoneManuelle + ZoneConforme;
}

public class ChequeStatusItem
{
    public byte Status { get; set; }
    public string Label { get; set; } = string.Empty;
    public int Count { get; set; }
}

public class ChequeStatusDistribution
{
    public List<ChequeStatusItem> Items { get; set; } = new();
    public int Total => Items.Sum(i => i.Count);
}
