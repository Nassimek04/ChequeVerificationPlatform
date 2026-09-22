using System.ComponentModel.DataAnnotations;

namespace ChequeVerification.Web.ViewModels.Cheques;

/// <summary>Row states shown in the batch preview table.</summary>
public static class BatchImportRowStatuses
{
    public const string Pret = "Pret";
    public const string ClientIntrouvable = "ClientIntrouvable";
    public const string CompteNonDetecte = "CompteNonDetecte";
    public const string ChequeNonDetecte = "ChequeNonDetecte";
    public const string MontantAVerifier = "MontantAVerifier";
    public const string ImageInvalide = "ImageInvalide";
    public const string ErreurOcr = "ErreurOcr";
    public const string Duplique = "Duplique";
    public const string IncoherenceDonnees = "IncoherenceDonnees";
    public const string Cmc7NonDetecte = "Cmc7NonDetecte";
    public const string Cmc7Invalide = "Cmc7Invalide";
    public const string VersoNonImportable = "VersoNonImportable";

    public static string Label(string status) => status switch
    {
        Pret => "Prêt",
        ClientIntrouvable => "Client introuvable",
        CompteNonDetecte => "N° compte non détecté",
        ChequeNonDetecte => "N° chèque non détecté",
        MontantAVerifier => "Montant à vérifier",
        ImageInvalide => "Image invalide",
        ErreurOcr => "Erreur OCR",
        Duplique => "Doublon",
        IncoherenceDonnees => "Incohérence données",
        Cmc7NonDetecte => "CMC7 non détecté",
        Cmc7Invalide => "CMC7 invalide",
        VersoNonImportable => "Verso / image non importable",
        _ => status
    };
}

/// <summary>Authoritative identity source for a batch row. CMC7 wins; Legacy
/// covers the pre-CMC7 synthetic harness (BLIND/CHQ) and nothing else.</summary>
public static class BatchImportSources
{
    public const string Cmc7 = "CMC7";
    public const string Legacy = "Legacy";
}

public class BatchImportRowViewModel
{
    public int Index { get; set; }

    /// <summary>Server-generated temp file name (GUID + extension). Never a user path.</summary>
    public string TempToken { get; set; } = string.Empty;

    public string OriginalFileName { get; set; } = string.Empty;

    /// <summary>Preview URL of the staged temp image.</summary>
    public string ImageUrl { get; set; } = string.Empty;

    public string? ChequeNumberRaw { get; set; }
    public string? ChequeNumber { get; set; }
    public string? AccountNumberRaw { get; set; }
    public string? AccountNumber { get; set; }

    /// <summary>Identity source: CMC7 (authoritative) or Legacy harness.</summary>
    public string MappingSource { get; set; } = BatchImportSources.Legacy;

    /// <summary>Raw CMC7 line as read by OCR (diagnostic evidence).</summary>
    public string? Cmc7Raw { get; set; }

    /// <summary>Canonical CMC7 cheque field (7 digits, no body prefix).</summary>
    public string? Cmc7Cheque { get; set; }

    /// <summary>Canonical CMC7 account (24 digits).</summary>
    public string? Cmc7Account { get; set; }

    /// <summary>Printed body cheque (diagnostic only, never authoritative).</summary>
    public string? PrintedCheque { get; set; }

    /// <summary>Printed body account (diagnostic only, never authoritative).</summary>
    public string? PrintedAccount { get; set; }

    /// <summary>CMC7/body coherence notes (suffix-compatible, exact, MISMATCH...).</summary>
    public string? CrossCheck { get; set; }

    public int? CustomerId { get; set; }
    public string? CustomerFullName { get; set; }
    public string? CustomerNumber { get; set; }

    public string? AmountRaw { get; set; }
    public decimal? Amount { get; set; }

    /// <summary>Prefilled editable amount text shown in the preview form (canonical fr format).</summary>
    public string? AmountEdit { get; set; }

    public string Status { get; set; } = BatchImportRowStatuses.ErreurOcr;
    public string StatusLabel => BatchImportRowStatuses.Label(Status);
    public string Message { get; set; } = string.Empty;
    public bool IsInsertable { get; set; }
}

public class BatchImportPreviewViewModel
{
    public List<BatchImportRowViewModel> Rows { get; set; } = new();
    public int TotalSelected => Rows.Count;
    public int ReadyCount => Rows.Count(r => r.IsInsertable);
    public int BlockedCount => Rows.Count(r => !r.IsInsertable);
    public string? GlobalError { get; set; }
}

public class BatchImportConfirmRowInput
{
    public string TempToken { get; set; } = string.Empty;
    public string OriginalFileName { get; set; } = string.Empty;
    public string? ChequeNumber { get; set; }
    public string? AccountNumber { get; set; }

    /// <summary>Identity source claimed by the preview; revalidated server-side.</summary>
    public string MappingSource { get; set; } = BatchImportSources.Legacy;

    /// <summary>User-corrected amount (French/Moroccan format accepted). Empty = keep OCR value.</summary>
    public string? CorrectedAmount { get; set; }

    /// <summary>OCR-parsed amount round-tripped as invariant string for fallback.</summary>
    public string? OcrAmount { get; set; }
}

public class BatchImportConfirmViewModel
{
    public List<BatchImportConfirmRowInput> Rows { get; set; } = new();
}

public class BatchImportResultRowViewModel
{
    public string? ChequeNumber { get; set; }
    public string? AccountNumber { get; set; }
    public string? CustomerFullName { get; set; }
    public decimal? Amount { get; set; }
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public int? ChequeId { get; set; }
}

public class BatchImportResultViewModel
{
    public int TotalSelected { get; set; }
    public int ImportedCount => Rows.Count(r => r.Success);
    public int IgnoredCount => Rows.Count(r => !r.Success);
    public List<BatchImportResultRowViewModel> Rows { get; set; } = new();
}

public class BatchImportUploadViewModel
{
    [Display(Name = "Images des chèques")]
    public List<IFormFile> Files { get; set; } = new();
}
