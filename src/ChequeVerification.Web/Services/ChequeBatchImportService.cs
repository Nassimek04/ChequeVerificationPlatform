using System.Globalization;
using System.Text.RegularExpressions;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Cheques;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Services;

/// <summary>
/// Phase 1 batch import. Reuses <see cref="IVerificationApiClient"/> OCR
/// (existing PaddleOCR pipeline) for import metadata only; signature
/// verification pipelines and decision policies are untouched.
/// </summary>
public class ChequeBatchImportService : IChequeBatchImportService
{
    public const int MaxBatchSize = 10;
    public const int MaxFileSizeBytes = 10 * 1024 * 1024;
    public const string TempSubFolder = "uploads/cheques/batch-temp";
    public const string FinalSubFolder = "uploads/cheques";

    private static readonly string[] AllowedExtensions = { ".jpg", ".jpeg", ".png" };
    private static readonly string[] AllowedContentTypes = { "image/jpeg", "image/png" };
    private static readonly byte[] PngMagic = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };

    private static readonly Regex BlindAccountPattern =
        new(@"BLIND\s*-\s*(\d{1,10})", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static readonly Regex ChequeRefPattern =
        new(@"CHQ\s*-\s*([A-Z0-9]+(?:\s*-\s*[A-Z0-9]+)*)", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private readonly ChequeVerificationDbContext _db;
    private readonly IVerificationApiClient _ocr;
    private readonly ILogger<ChequeBatchImportService> _logger;

    public ChequeBatchImportService(
        ChequeVerificationDbContext db,
        IVerificationApiClient ocr,
        ILogger<ChequeBatchImportService> logger)
    {
        _db = db;
        _ocr = ocr;
        _logger = logger;
    }

    // ------------------------------------------------------------------
    // Normalization + amount parsing (public static so unit tests target them).
    // ------------------------------------------------------------------

    /// <summary>
    /// Trim, uppercase, strip harmless OCR spacing, preserve '-'.
    /// " blind - 0001 " -&gt; "BLIND-0001".
    /// </summary>
    public static string? NormalizeAccountNumber(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        var upper = raw.Trim().ToUpperInvariant();
        // Remove every whitespace char (spaces, NBSP, tabs) — harmless OCR
        // spacing around '-' disappears, meaningful '-' is preserved.
        upper = Regex.Replace(upper, @"\s+", string.Empty);
        return string.IsNullOrEmpty(upper) ? null : upper;
    }

    /// <summary>Trim, uppercase, normalize spacing, preserve '-'.</summary>
    public static string? NormalizeChequeNumber(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        var upper = raw.Trim().ToUpperInvariant();
        upper = Regex.Replace(upper, @"\s+", string.Empty);
        return string.IsNullOrEmpty(upper) ? null : upper;
    }

    /// <summary>
    /// Parses Moroccan/French formatted amounts to decimal.
    /// Accepts "12 500,00", "12.500,00", "12500.00", "12 500 MAD".
    /// Returns false (never guesses) when parsing fails.
    /// </summary>
    public static bool TryParseAmount(string? raw, out decimal amount)
    {
        amount = 0;
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        var s = raw.Trim().ToUpperInvariant();
        // Strip currency tokens (MAD/DH/DHS/DIRHAM(S)/DA) and ALL whitespace
        // (regular, NBSP U+00A0, narrow NBSP U+202F from fr-FR formatting).
        s = Regex.Replace(s, @"(DIRHAMS?|DHS?|MAD|\bDA\b)", string.Empty);
        s = Regex.Replace(s, @"\s+", string.Empty).Replace("'", string.Empty).Trim();
        s = s.Trim('.', ',');
        if (s.Length == 0)
        {
            return false;
        }

        if (s.Contains(',') && s.Contains('.'))
        {
            // Rightmost separator is the decimal one.
            if (s.LastIndexOf(',') > s.LastIndexOf('.'))
            {
                s = s.Replace(".", string.Empty).Replace(',', '.'); // fr: 12.500,00
            }
            else
            {
                s = s.Replace(",", string.Empty); // us: 12,500.00
            }
        }
        else if (s.Contains(','))
        {
            s = s.Replace(".", string.Empty).Replace(',', '.');
        }
        else if (s.Contains('.'))
        {
            // Multiple dots => thousands (1.000.000.00); single dot kept.
            var parts = s.Split('.');
            if (parts.Length > 2)
            {
                s = string.Concat(parts[..^1]) + "." + parts[^1];
            }
        }

        // After normalization only digits, one dot, optional leading minus.
        if (!Regex.IsMatch(s, @"^-?\d+(\.\d{1,2})?$"))
        {
            return false;
        }

        if (!decimal.TryParse(s, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint,
                CultureInfo.InvariantCulture, out amount))
        {
            return false;
        }

        if (amount < 0 || amount > 999_999_999)
        {
            return false;
        }

        amount = Math.Round(amount, 2);
        return true;
    }

    /// <summary>Canonical display: "12 500,00 MAD".</summary>
    public static string FormatMad(decimal amount)
        => amount.ToString("N2", CultureInfo.GetCultureInfo("fr-FR")) + " MAD";

    // ------------------------------------------------------------------
    // CMC7-authoritative mapping (supervisor real-cheque dataset).
    // Canonical: cheque = 7-digit CMC7 field (no "253" prefix fabrication);
    // account = 24-digit CMC7 RIB. Printed body values are diagnostics only.
    // ------------------------------------------------------------------

    private static readonly Regex Cmc7ChequeShape =
        new(@"^\d{7}$", RegexOptions.Compiled);

    private static readonly Regex Cmc7AccountShape =
        new(@"^\d{24}$", RegexOptions.Compiled);

    private static readonly string[] VersoMarkers =
        { "AVIS IMPORTANT", "TRG3", "CODE DE COMMERCE", "15-95" };

    /// <summary>Structural gate on parser output (7 digits / 24 digits).</summary>
    public static bool IsCmc7ChequeShape(string? v)
        => v != null && Cmc7ChequeShape.IsMatch(v);

    /// <summary>Structural gate on parser output (7 digits / 24 digits).</summary>
    public static bool IsCmc7AccountShape(string? v)
        => v != null && Cmc7AccountShape.IsMatch(v);

    internal sealed record ResolvedMapping(
        string Source,
        string? Cheque,
        string? Account,
        string? Cmc7Raw,
        string? Cmc7Cheque,
        string? Cmc7Account,
        string? PrintedCheque,
        string? PrintedAccount,
        string? CrossCheck,
        string? Cmc7Error,
        bool IsProbableVerso);

    /// <summary>
    /// Pure CMC7-first resolution shared by preview and confirm revalidation.
    /// Valid CMC7 is authoritative; legacy BLIND/CHQ markers (pre-CMC7 test
    /// harness only) are preserved; anything else is unresolved and the
    /// printed body is NEVER substituted.
    /// </summary>
    internal static ResolvedMapping ResolveMapping(Dtos.VerificationApi.ChequeOcrResponseDto ocr)
    {
        var fields = ocr.Fields;
        var searchText = ocr.FullText ?? string.Empty;
        if (ocr.Lines != null && ocr.Lines.Count > 0)
        {
            searchText += "\n" + string.Join("\n", ocr.Lines.Select(l => l.Text));
        }

        var verso = fields?.IsProbableVerso == true
            || VersoMarkers.Any(m => searchText.ToUpperInvariant().Contains(m));

        // 1. CMC7 authoritative path (parser output + structural gate).
        if (fields?.Cmc7Valid == true
            && IsCmc7ChequeShape(fields.Cmc7ChequeNumber)
            && IsCmc7AccountShape(fields.Cmc7AccountNumber))
        {
            return new ResolvedMapping(
                BatchImportSources.Cmc7,
                NormalizeChequeNumber(fields.Cmc7ChequeNumber),
                NormalizeAccountNumber(fields.Cmc7AccountNumber),
                fields.Cmc7Raw,
                NormalizeChequeNumber(fields.Cmc7ChequeNumber),
                NormalizeAccountNumber(fields.Cmc7AccountNumber),
                NormalizeChequeNumber(fields.PrintedChequeNumber),
                NormalizeAccountNumber(fields.PrintedAccountNumber),
                fields.Cmc7CrossCheck,
                null,
                false); // valid CMC7 is never overridden by verso heuristics
        }

        // 2. Legacy pre-CMC7 harness path (BLIND-XXXX / CHQ-... ONLY).
        // Generic OCR values (phone numbers, amounts, body fragments) must
        // never leak in here: the business rule forbids body substitution
        // when CMC7 is missing/invalid.
        var chequeRaw = fields?.ChequeNumber;
        var accountRaw = fields?.AccountNumber;
        if (string.IsNullOrWhiteSpace(accountRaw))
        {
            var m = BlindAccountPattern.Match(searchText);
            if (m.Success)
            {
                accountRaw = "BLIND-" + m.Groups[1].Value;
            }
        }

        if (string.IsNullOrWhiteSpace(chequeRaw))
        {
            var m = ChequeRefPattern.Match(searchText);
            if (m.Success)
            {
                chequeRaw = "CHQ-" + Regex.Replace(m.Groups[1].Value.ToUpperInvariant(), @"\s+", string.Empty);
            }
        }

        var legacyCheque = NormalizeChequeNumber(chequeRaw);
        var legacyAccount = NormalizeAccountNumber(accountRaw);
        var isLegacyCheque = legacyCheque != null && legacyCheque.StartsWith("CHQ-", StringComparison.Ordinal);
        var isLegacyAccount = legacyAccount != null
            && Regex.IsMatch(legacyAccount, @"^BLIND-\d{1,10}$", RegexOptions.IgnoreCase);
        if (isLegacyCheque || isLegacyAccount)
        {
            return new ResolvedMapping(
                BatchImportSources.Legacy,
                legacyCheque, legacyAccount,
                fields?.Cmc7Raw, null, null,
                NormalizeChequeNumber(fields?.PrintedChequeNumber),
                NormalizeAccountNumber(fields?.PrintedAccountNumber),
                fields?.Cmc7CrossCheck, fields?.Cmc7Error, verso);
        }

        // 3. Unresolved: verso, invalid CMC7, or nothing detected.
        // Printed body values are deliberately NOT substituted.
        return new ResolvedMapping(
            BatchImportSources.Legacy, null, null,
            fields?.Cmc7Raw, null, null,
            NormalizeChequeNumber(fields?.PrintedChequeNumber),
            NormalizeAccountNumber(fields?.PrintedAccountNumber),
            fields?.Cmc7CrossCheck, fields?.Cmc7Error, verso);
    }

    // ------------------------------------------------------------------
    // Preview
    // ------------------------------------------------------------------

    public async Task<BatchImportPreviewViewModel> BuildPreviewAsync(
        IList<IFormFile> files, string webRootPath, CancellationToken cancellationToken = default)
    {
        var preview = new BatchImportPreviewViewModel();

        if (files == null || files.Count == 0)
        {
            preview.GlobalError = "Sélectionnez au moins une image de chèque.";
            return preview;
        }

        if (files.Count > MaxBatchSize)
        {
            preview.GlobalError = $"Maximum {MaxBatchSize} images par lot.";
            return preview;
        }

        // Snapshot existing cheque numbers once (exact match after normalization).
        var existingNumbers = new HashSet<string>(StringComparer.Ordinal);
        try
        {
            var all = await _db.Cheques.AsNoTracking()
                .Select(c => c.ChequeNumber).ToListAsync(cancellationToken);
            foreach (var n in all)
            {
                var norm = NormalizeChequeNumber(n);
                if (norm != null)
                {
                    existingNumbers.Add(norm);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Impossible de charger les numéros de chèques existants.");
        }

        var seenInBatch = new HashSet<string>(StringComparer.Ordinal);
        var tempDir = Path.Combine(webRootPath, TempSubFolder.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(tempDir);

        for (var i = 0; i < files.Count; i++)
        {
            var file = files[i];
            var row = new BatchImportRowViewModel
            {
                Index = i,
                OriginalFileName = Path.GetFileName(file.FileName)
            };

            var bytes = await ReadAndValidateImageAsync(file, row);
            if (bytes == null)
            {
                preview.Rows.Add(row); // ImageInvalide already set
                continue;
            }

            // Stage to temp file (server-side GUID name).
            var ext = Path.GetExtension(file.FileName).ToLowerInvariant();
            var token = $"{Guid.NewGuid():N}{ext}";
            try
            {
                await System.IO.File.WriteAllBytesAsync(Path.Combine(tempDir, token), bytes, cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Échec de la mise en attente du fichier {Name}.", row.OriginalFileName);
                row.Status = BatchImportRowStatuses.ImageInvalide;
                row.Message = "Impossible de préparer l'image.";
                preview.Rows.Add(row);
                continue;
            }

            row.TempToken = token;
            row.ImageUrl = $"/{TempSubFolder}/{token}";

            await FillOcrRowAsync(row, bytes, Path.GetFileName(file.FileName),
                ResolveContentType(ext), existingNumbers, seenInBatch, cancellationToken);

            preview.Rows.Add(row);
        }

        return preview;
    }

    private static async Task<byte[]?> ReadAndValidateImageAsync(IFormFile file, BatchImportRowViewModel row)
    {
        if (file == null || file.Length <= 0)
        {
            row.Status = BatchImportRowStatuses.ImageInvalide;
            row.Message = "Fichier vide.";
            return null;
        }

        if (file.Length > MaxFileSizeBytes)
        {
            row.Status = BatchImportRowStatuses.ImageInvalide;
            row.Message = "Fichier trop volumineux (maximum 10 Mo).";
            return null;
        }

        var ext = Path.GetExtension(file.FileName).ToLowerInvariant();
        if (!AllowedExtensions.Contains(ext))
        {
            row.Status = BatchImportRowStatuses.ImageInvalide;
            row.Message = "Format non supporté. Formats acceptés : .jpg, .jpeg, .png.";
            return null;
        }

        if (!AllowedContentTypes.Contains(file.ContentType.ToLowerInvariant()))
        {
            row.Status = BatchImportRowStatuses.ImageInvalide;
            row.Message = "Type de contenu non autorisé (image/jpeg, image/png).";
            return null;
        }

        byte[] bytes;
        try
        {
            await using var ms = new MemoryStream();
            await file.CopyToAsync(ms);
            bytes = ms.ToArray();
        }
        catch
        {
            row.Status = BatchImportRowStatuses.ImageInvalide;
            row.Message = "Impossible de lire le fichier.";
            return null;
        }

        if (bytes.Length == 0 || bytes.Length > MaxFileSizeBytes)
        {
            row.Status = BatchImportRowStatuses.ImageInvalide;
            row.Message = "Taille du fichier invalide.";
            return null;
        }

        // Actual image decoding check: magic bytes + dimension sanity.
        if (!TryValidateImageBytes(bytes, ext, out var imageError))
        {
            row.Status = BatchImportRowStatuses.ImageInvalide;
            row.Message = imageError;
            return null;
        }

        return bytes;
    }

    /// <summary>
    /// Header-level image validation without native dependencies:
    /// PNG magic + IHDR dimensions, JPEG SOI marker + minimal size.
    /// </summary>
    internal static bool TryValidateImageBytes(byte[] bytes, string extension, out string error)
    {
        error = "L'image ne peut pas être décodée.";
        if (bytes.Length < 16)
        {
            return false;
        }

        if (extension is ".png")
        {
            for (var i = 0; i < PngMagic.Length; i++)
            {
                if (bytes[i] != PngMagic[i])
                {
                    return false;
                }
            }

            // IHDR width/height at offsets 16..23 (big-endian).
            if (bytes.Length >= 24)
            {
                var w = (bytes[16] << 24) | (bytes[17] << 16) | (bytes[18] << 8) | bytes[19];
                var h = (bytes[20] << 24) | (bytes[21] << 16) | (bytes[22] << 8) | bytes[23];
                if ((w < 32 || h < 32 || w > 8000 || h > 8000) && w > 0 && h > 0)
                {
                    error = "Dimensions d'image invalides pour l'OCR.";
                    return false;
                }
            }

            error = string.Empty;
            return true;
        }

        // JPEG
        if (bytes[0] == 0xFF && bytes[1] == 0xD8)
        {
            if (bytes.Length < 64)
            {
                return false;
            }

            error = string.Empty;
            return true;
        }

        return false;
    }

    private static string ResolveContentType(string extension) =>
        extension == ".png" ? "image/png" : "image/jpeg";

    private async Task FillOcrRowAsync(
        BatchImportRowViewModel row, byte[] bytes, string fileName, string contentType,
        HashSet<string> existingNumbers, HashSet<string> seenInBatch,
        CancellationToken cancellationToken)
    {
        Dtos.VerificationApi.ChequeOcrResponseDto? ocr;
        try
        {
            await using var stream = new MemoryStream(bytes, writable: false);
            ocr = await _ocr.OcrChequeAsync(stream, fileName, contentType, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "OCR indisponible pour {Name}.", row.OriginalFileName);
            ocr = null;
        }

        if (ocr == null || !ocr.Success)
        {
            row.Status = BatchImportRowStatuses.ErreurOcr;
            row.Message = string.IsNullOrWhiteSpace(ocr?.Message)
                ? "Le service OCR est indisponible ou n'a pas répondu correctement."
                : ocr.Message;
            return;
        }

        // 1. CMC7-first identity resolution (shared with confirm revalidation).
        // Valid CMC7 is authoritative; the printed body is diagnostic only
        // and is NEVER substituted when CMC7 is missing/invalid.
        var mapping = ResolveMapping(ocr);
        row.MappingSource = mapping.Source;
        row.Cmc7Raw = mapping.Cmc7Raw;
        row.Cmc7Cheque = mapping.Cmc7Cheque;
        row.Cmc7Account = mapping.Cmc7Account;
        row.PrintedCheque = mapping.PrintedCheque;
        row.PrintedAccount = mapping.PrintedAccount;
        row.CrossCheck = mapping.CrossCheck;

        if (mapping.Source == BatchImportSources.Cmc7)
        {
            row.ChequeNumberRaw = ocr.Fields?.Cmc7ChequeNumber;
            row.AccountNumberRaw = ocr.Fields?.Cmc7AccountNumber;
            row.ChequeNumber = mapping.Cheque;
            row.AccountNumber = mapping.Account;

            // Body cross-check: explicit disagreement with a VALID CMC7
            // blocks the row (human review). Absent body = no check.
            if (mapping.CrossCheck != null && mapping.CrossCheck.Contains("MISMATCH", StringComparison.Ordinal))
            {
                row.Status = BatchImportRowStatuses.IncoherenceDonnees;
                row.Message = $"Incohérence CMC7 / corps imprimé : CMC7 chèque {row.ChequeNumber}, " +
                    $"compte {row.AccountNumber} — corps [{mapping.CrossCheck}].";
                return;
            }
        }
        else if (!string.IsNullOrEmpty(mapping.Cheque) || !string.IsNullOrEmpty(mapping.Account))
        {
            // Legacy harness path only (BLIND/CHQ shapes, see ResolveMapping).
            row.ChequeNumberRaw = ocr.Fields?.ChequeNumber;
            row.AccountNumberRaw = ocr.Fields?.AccountNumber;
            row.ChequeNumber = mapping.Cheque;
            row.AccountNumber = mapping.Account;
        }
        else if (mapping.IsProbableVerso)
        {
            row.Status = BatchImportRowStatuses.VersoNonImportable;
            row.Message = "Verso détecté / image non importable comme recto de chèque.";
            return;
        }
        else if (!string.IsNullOrEmpty(mapping.Cmc7Raw))
        {
            row.Status = BatchImportRowStatuses.Cmc7Invalide;
            row.Message = $"CMC7 invalide — {mapping.Cmc7Error ?? "structure non conforme"}. " +
                "Le corps imprimé ne peut pas s'y substituer.";
            return;
        }
        else
        {
            row.Status = BatchImportRowStatuses.Cmc7NonDetecte;
            row.Message = "CMC7 non détecté par l'OCR. Le corps imprimé ne peut pas s'y substituer.";
            return;
        }

        var amountRaw = ocr.Fields?.AmountNumeric;
        var amountTextFallback = ocr.Fields?.AmountText;

        // 2. Amount: prefer numeric field, fallback to amount text. Never guess.
        decimal parsed = 0;
        var hasAmount = !string.IsNullOrWhiteSpace(amountRaw) && TryParseAmount(amountRaw, out parsed);
        if (!hasAmount && !string.IsNullOrWhiteSpace(amountTextFallback))
        {
            amountRaw = amountTextFallback;
            hasAmount = TryParseAmount(amountRaw, out parsed);
        }
        else if (hasAmount)
        {
            row.AmountRaw = amountRaw;
        }
        else
        {
            row.AmountRaw = amountRaw ?? amountTextFallback;
        }

        if (hasAmount)
        {
            row.Amount = parsed;
            row.AmountEdit = parsed.ToString("N2", CultureInfo.GetCultureInfo("fr-FR"));
        }
        else
        {
            row.Amount = null;
            row.AmountEdit = null;
        }

        // 3. Cheque number presence + uniqueness (DB + within batch).
        if (string.IsNullOrEmpty(row.ChequeNumber))
        {
            row.Status = BatchImportRowStatuses.ChequeNonDetecte;
            row.Message = "N° chèque non détecté par l'OCR.";
            return;
        }

        if (existingNumbers.Contains(row.ChequeNumber) || !seenInBatch.Add(row.ChequeNumber))
        {
            row.Status = BatchImportRowStatuses.Duplique;
            row.Message = $"Le chèque {row.ChequeNumber} existe déjà.";
            return;
        }

        // 4. Exact customer mapping (no fuzzy name matching, no creation).
        if (string.IsNullOrEmpty(row.AccountNumber))
        {
            row.Status = BatchImportRowStatuses.CompteNonDetecte;
            row.Message = "N° compte non détecté par l'OCR.";
            return;
        }

        var matches = await FindCustomersByAccountAsync(row.AccountNumber, cancellationToken);
        if (matches.Count == 0)
        {
            row.Status = BatchImportRowStatuses.ClientIntrouvable;
            row.Message = $"Aucun client trouvé pour le compte {row.AccountNumber}.";
            return;
        }

        if (matches.Count > 1)
        {
            row.Status = BatchImportRowStatuses.IncoherenceDonnees;
            row.Message = $"Plusieurs clients partagent le compte {row.AccountNumber}.";
            return;
        }

        row.CustomerId = matches[0].CustomerId;
        row.CustomerFullName = matches[0].FullName;
        row.CustomerNumber = matches[0].CustomerNumber;

        // 5. Amount gate: visible + correctable; missing amount blocks insert.
        if (!hasAmount)
        {
            row.Status = BatchImportRowStatuses.MontantAVerifier;
            row.Message = "Montant non détecté — corrigez-le avant confirmation.";
            return;
        }

        row.Status = BatchImportRowStatuses.Pret;
        row.Message = "Compte client trouvé.";
        row.IsInsertable = true;
    }

    private async Task<List<Customer>> FindCustomersByAccountAsync(
        string normalizedAccount, CancellationToken cancellationToken)
    {
        // Fast path: exact match (SQL collation is typically case-insensitive;
        // stored values are already normalized like BLIND-0001).
        var direct = await _db.Customers.AsNoTracking()
            .Where(c => c.AccountNumber == normalizedAccount)
            .ToListAsync(cancellationToken);
        if (direct.Count > 0)
        {
            return direct;
        }

        // Fallback: normalized in-memory comparison in case stored values
        // contain harmless spacing variants.
        var all = await _db.Customers.AsNoTracking()
            .Select(c => new { c.CustomerId, c.CustomerNumber, c.FullName, c.AccountNumber })
            .ToListAsync(cancellationToken);
        var ids = all
            .Where(c => NormalizeAccountNumber(c.AccountNumber) == normalizedAccount)
            .Select(c => c.CustomerId)
            .ToList();
        if (ids.Count == 0)
        {
            return new List<Customer>();
        }

        return await _db.Customers.AsNoTracking()
            .Where(c => ids.Contains(c.CustomerId))
            .ToListAsync(cancellationToken);
    }

    // ------------------------------------------------------------------
    // Confirm (partial success)
    // ------------------------------------------------------------------

    public async Task<BatchImportResultViewModel> ConfirmImportAsync(
        BatchImportConfirmViewModel model, int userId, string webRootPath,
        CancellationToken cancellationToken = default)
    {
        var result = new BatchImportResultViewModel
        {
            TotalSelected = model?.Rows?.Count ?? 0
        };

        if (model?.Rows == null || model.Rows.Count == 0)
        {
            return result;
        }

        var tempDir = Path.Combine(webRootPath, TempSubFolder.Replace('/', Path.DirectorySeparatorChar));
        var finalDir = Path.Combine(webRootPath, FinalSubFolder.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(finalDir);

        // Fresh duplicate snapshot for this confirmation.
        var existingNumbers = new HashSet<string>(StringComparer.Ordinal);
        var allNumbers = await _db.Cheques.AsNoTracking()
            .Select(c => c.ChequeNumber).ToListAsync(cancellationToken);
        foreach (var n in allNumbers)
        {
            var norm = NormalizeChequeNumber(n);
            if (norm != null)
            {
                existingNumbers.Add(norm);
            }
        }

        var seenInBatch = new HashSet<string>(StringComparer.Ordinal);

        foreach (var input in model.Rows)
        {
            var outcome = new BatchImportResultRowViewModel
            {
                ChequeNumber = NormalizeChequeNumber(input.ChequeNumber),
                AccountNumber = NormalizeAccountNumber(input.AccountNumber)
            };

            try
            {
                // Re-resolve temp file safely (no user-supplied paths).
                var token = Path.GetFileName(input.TempToken ?? string.Empty);
                if (string.IsNullOrEmpty(token) || token != input.TempToken ||
                    !AllowedExtensions.Contains(Path.GetExtension(token).ToLowerInvariant()))
                {
                    outcome.Success = false;
                    outcome.Message = "Référence d'image invalide.";
                    result.Rows.Add(outcome);
                    continue;
                }

                var tempPath = Path.Combine(tempDir, token);
                if (!System.IO.File.Exists(tempPath))
                {
                    outcome.Success = false;
                    outcome.Message = "Image en attente introuvable — relancez l'import.";
                    result.Rows.Add(outcome);
                    continue;
                }

                // Server-side revalidation: re-read the STAGED image through
                // OCR and re-resolve identity. Hidden preview values
                // (CustomerId, numbers, source) are never trusted.
                ResolvedMapping fresh;
                try
                {
                    var stagedBytes = await System.IO.File.ReadAllBytesAsync(tempPath, cancellationToken);
                    await using var ocrStream = new MemoryStream(stagedBytes, writable: false);
                    var freshOcr = await _ocr.OcrChequeAsync(
                        ocrStream, input.OriginalFileName,
                        ResolveContentType(Path.GetExtension(token).ToLowerInvariant()),
                        cancellationToken);
                    if (freshOcr == null || !freshOcr.Success)
                    {
                        outcome.Success = false;
                        outcome.Message = "Relecture OCR impossible — relancez l'import.";
                        result.Rows.Add(outcome);
                        continue;
                    }

                    fresh = ResolveMapping(freshOcr);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Revalidation OCR impossible pour {Name}.", input.OriginalFileName);
                    outcome.Success = false;
                    outcome.Message = "Relecture OCR impossible — relancez l'import.";
                    result.Rows.Add(outcome);
                    continue;
                }

                var claimedSource = string.IsNullOrEmpty(input.MappingSource)
                    ? BatchImportSources.Legacy
                    : input.MappingSource;
                if (!string.Equals(fresh.Source, claimedSource, StringComparison.Ordinal)
                    || !string.Equals(fresh.Cheque, NormalizeChequeNumber(input.ChequeNumber), StringComparison.Ordinal)
                    || !string.Equals(fresh.Account, NormalizeAccountNumber(input.AccountNumber), StringComparison.Ordinal))
                {
                    outcome.ChequeNumber = NormalizeChequeNumber(input.ChequeNumber);
                    outcome.AccountNumber = NormalizeAccountNumber(input.AccountNumber);
                    outcome.Success = false;
                    outcome.Message = "Données d'aperçu incohérentes avec la relecture — relancez l'import.";
                    result.Rows.Add(outcome);
                    continue;
                }

                if (fresh.Source == BatchImportSources.Cmc7
                    && fresh.CrossCheck != null
                    && fresh.CrossCheck.Contains("MISMATCH", StringComparison.Ordinal))
                {
                    outcome.ChequeNumber = fresh.Cheque;
                    outcome.AccountNumber = fresh.Account;
                    outcome.Success = false;
                    outcome.Message = "Incohérence CMC7 / corps imprimé — import refusé.";
                    result.Rows.Add(outcome);
                    continue;
                }

                if (fresh.Source != BatchImportSources.Cmc7
                    && string.IsNullOrEmpty(fresh.Cheque)
                    && string.IsNullOrEmpty(fresh.Account))
                {
                    outcome.ChequeNumber = NormalizeChequeNumber(input.ChequeNumber);
                    outcome.AccountNumber = NormalizeAccountNumber(input.AccountNumber);
                    outcome.Success = false;
                    outcome.Message = fresh.IsProbableVerso
                        ? "Verso détecté / image non importable comme recto de chèque."
                        : "Identité non résolue à la relecture — relancez l'import.";
                    result.Rows.Add(outcome);
                    continue;
                }

                // From here on, ONLY freshly revalidated values are used.
                outcome.ChequeNumber = fresh.Cheque;
                outcome.AccountNumber = fresh.Account;

                if (string.IsNullOrEmpty(outcome.ChequeNumber))
                {
                    outcome.Success = false;
                    outcome.Message = "N° chèque manquant.";
                    result.Rows.Add(outcome);
                    continue;
                }

                if (existingNumbers.Contains(outcome.ChequeNumber) || !seenInBatch.Add(outcome.ChequeNumber))
                {
                    outcome.Success = false;
                    outcome.Message = $"Le chèque {outcome.ChequeNumber} existe déjà.";
                    result.Rows.Add(outcome);
                    continue;
                }

                if (string.IsNullOrEmpty(outcome.AccountNumber))
                {
                    outcome.Success = false;
                    outcome.Message = "N° compte manquant.";
                    result.Rows.Add(outcome);
                    continue;
                }

                // Server-side customer re-mapping (hidden CustomerId is not trusted).
                var matches = await FindCustomersByAccountAsync(outcome.AccountNumber, cancellationToken);
                if (matches.Count == 0)
                {
                    outcome.Success = false;
                    outcome.Message = $"Aucun client trouvé pour le compte {outcome.AccountNumber}.";
                    result.Rows.Add(outcome);
                    continue;
                }

                if (matches.Count > 1)
                {
                    outcome.Success = false;
                    outcome.Message = $"Plusieurs clients partagent le compte {outcome.AccountNumber}.";
                    result.Rows.Add(outcome);
                    continue;
                }

                var customer = matches[0];
                outcome.CustomerFullName = customer.FullName;

                // Amount: corrected value wins, else OCR round-trip. Never fabricate.
                decimal amountValue = 0;
                var hasAmount = false;
                if (!string.IsNullOrWhiteSpace(input.CorrectedAmount))
                {
                    hasAmount = TryParseAmount(input.CorrectedAmount, out amountValue);
                    if (!hasAmount)
                    {
                        outcome.Success = false;
                        outcome.Message = "Montant corrigé invalide.";
                        outcome.Amount = null;
                        result.Rows.Add(outcome);
                        continue;
                    }
                }
                else if (!string.IsNullOrWhiteSpace(input.OcrAmount) &&
                         decimal.TryParse(input.OcrAmount, NumberStyles.Any,
                             CultureInfo.InvariantCulture, out var ocrAmt))
                {
                    amountValue = ocrAmt;
                    hasAmount = true;
                }

                if (!hasAmount)
                {
                    outcome.Success = false;
                    outcome.Message = "Montant à vérifier avant import.";
                    result.Rows.Add(outcome);
                    continue;
                }

                outcome.Amount = amountValue;

                // Persist image using existing storage conventions (GUID name).
                var ext = Path.GetExtension(token).ToLowerInvariant();
                var finalName = $"{Guid.NewGuid():N}{ext}";
                var finalFullPath = Path.Combine(finalDir, finalName);
                System.IO.File.Copy(tempPath, finalFullPath);
                var imagePath = $"/{FinalSubFolder}/{finalName}";

                var cheque = new Cheque
                {
                    CustomerId = customer.CustomerId,
                    ImportedByUserId = userId,
                    ChequeNumber = outcome.ChequeNumber,
                    Amount = amountValue,
                    IssueDate = null,
                    ImagePath = imagePath,
                    Status = 1, // En attente
                    UploadedAt = DateTime.UtcNow
                };

                try
                {
                    _db.Cheques.Add(cheque);
                    await _db.SaveChangesAsync(cancellationToken);

                    _db.AuditLogs.Add(new AuditLog
                    {
                        UserId = userId,
                        Action = "IMPORT_CHEQUE",
                        EntityName = nameof(Cheque),
                        EntityId = cheque.ChequeId,
                        Description = $"Import par lot du chèque {cheque.ChequeNumber} pour le client n° {customer.CustomerId}",
                        CreatedAt = DateTime.UtcNow
                    });
                    await _db.SaveChangesAsync(cancellationToken);
                }
                catch
                {
                    TryDeleteStagedFile(finalFullPath, finalDir);
                    throw;
                }

                // Per-row temp cleanup (best effort, app-owned dir only).
                TryDeleteStagedFile(tempPath, tempDir);

                outcome.Success = true;
                outcome.Message = "Importé.";
                outcome.ChequeId = cheque.ChequeId;
            }
            catch (DbUpdateException ex)
            {
                _logger.LogWarning(ex, "Conflit d'insertion pour le chèque {Number}.", outcome.ChequeNumber);
                outcome.Success = false;
                outcome.Message = "Conflit d'insertion (doublon probable).";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Échec de l'import du chèque {Number}.", outcome.ChequeNumber);
                outcome.Success = false;
                outcome.Message = "Erreur lors de l'import.";
            }

            result.Rows.Add(outcome);
        }

        return result;
    }

    /// <summary>
    /// Ownership containment for cleanup: staging/cleanup must operate only
    /// inside application-owned directories (webRoot temp/final). Anything
    /// resolving outside the owning root is never touched, so user-selected
    /// source files (which the server only ever sees as upload streams) can
    /// never be mutated by import cleanup.
    /// </summary>
    internal static bool IsOwnedPath(string? path, string? ownedRoot)
    {
        if (string.IsNullOrEmpty(path) || string.IsNullOrEmpty(ownedRoot))
        {
            return false;
        }

        try
        {
            var full = Path.GetFullPath(path);
            var root = Path.GetFullPath(ownedRoot).TrimEnd(Path.DirectorySeparatorChar)
                + Path.DirectorySeparatorChar;
            return full.StartsWith(root, StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private static void TryDeleteStagedFile(string path, string ownedRoot)
    {
        try
        {
            if (IsOwnedPath(path, ownedRoot) && System.IO.File.Exists(path))
            {
                System.IO.File.Delete(path);
            }
        }
        catch
        {
            // Best-effort cleanup only.
        }
    }
}
