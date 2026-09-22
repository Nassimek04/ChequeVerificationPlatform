using System.Net.Http.Json;
using System.Text.Json;
using System.Text.RegularExpressions;
using ChequeVerification.Web.Dtos.VerificationApi;
using ChequeVerification.Web.Services.Interfaces;

namespace ChequeVerification.Web.Services;

public class VerificationApiClient : IVerificationApiClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<VerificationApiClient> _logger;

    // Field names carrying base64 images in FastAPI responses. Their values are
    // never logged in full.
    private static readonly HashSet<string> Base64Fields = new(StringComparer.OrdinalIgnoreCase)
    {
        "signature_image_base64",
        "original_with_roi_base64",
        "roi_image_base64",
        "mask_image_base64",
        "components_all_base64",
        "components_rejected_base64",
        "components_image_base64",
        "group_image_base64",
        "groups_image_base64",
        "refined_group_image_base64",
        "fallback_candidates_image_base64",
    };

    public VerificationApiClient(HttpClient httpClient, ILogger<VerificationApiClient> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
    }

    public async Task<HealthResponseDto?> GetHealthAsync(CancellationToken cancellationToken = default)
    {
        const string endpoint = "/api/health";
        return await CallAsync(
            endpoint,
            async ct =>
            {
                var response = await _httpClient.GetAsync(endpoint, ct);
                return await HandleResponseAsync<HealthResponseDto>(endpoint, response, ct);
            },
            cancellationToken);
    }

    public async Task<ImageAnalysisResponseDto?> AnalyzeChequeImageAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
    {
        const string endpoint = "/api/images/analyze";
        return await CallAsync(
            endpoint,
            async ct =>
            {
                using var content = new MultipartFormDataContent();
                using var fileContent = new StreamContent(imageStream);
                fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(contentType);
                content.Add(fileContent, "file", fileName);

                var response = await _httpClient.PostAsync(endpoint, content, ct);
                return await HandleResponseAsync<ImageAnalysisResponseDto>(endpoint, response, ct);
            },
            cancellationToken);
    }

    public async Task<SignatureExtractionResponseDto?> ExtractSignatureAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
    {
        const string endpoint = "/api/signatures/extract";
        return await CallAsync(
            endpoint,
            async ct =>
            {
                using var content = new MultipartFormDataContent();
                using var fileContent = new StreamContent(imageStream);
                fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(contentType);
                content.Add(fileContent, "file", fileName);

                var response = await _httpClient.PostAsync(endpoint, content, ct);
                return await HandleResponseAsync<SignatureExtractionResponseDto>(endpoint, response, ct);
            },
            cancellationToken);
    }

    public async Task<SignatureDebugResponseDto?> DebugSignatureExtractionAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
    {
        const string endpoint = "/api/signatures/debug";
        return await CallAsync(
            endpoint,
            async ct =>
            {
                using var content = new MultipartFormDataContent();
                using var fileContent = new StreamContent(imageStream);
                fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(contentType);
                content.Add(fileContent, "file", fileName);

                var response = await _httpClient.PostAsync(endpoint, content, ct);
                return await HandleResponseAsync<SignatureDebugResponseDto>(endpoint, response, ct);
            },
            cancellationToken);
    }

    public async Task<SignatureComparisonResponseDto?> CompareSignaturesAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default)
    {
        const string endpoint = "/api/signatures/compare";
        return await CallAsync(
            endpoint,
            async ct =>
            {
                using var content = new MultipartFormDataContent();
                using var extractedContent = new StreamContent(extractedSignature);
                extractedContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(extractedContentType);
                content.Add(extractedContent, "extracted_file", extractedFileName);

                using var referenceContent = new StreamContent(referenceSignature);
                referenceContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(referenceContentType);
                content.Add(referenceContent, "reference_file", referenceFileName);

                var response = await _httpClient.PostAsync(endpoint, content, ct);
                return await HandleResponseAsync<SignatureComparisonResponseDto>(endpoint, response, ct);
            },
            cancellationToken);
    }

    public async Task<ChequeOcrResponseDto?> OcrChequeAsync(Stream imageStream, string fileName, string contentType, CancellationToken cancellationToken = default)
    {
        const string endpoint = "/api/cheques/ocr";
        return await CallAsync(
            endpoint,
            async ct =>
            {
                using var content = new MultipartFormDataContent();
                using var fileContent = new StreamContent(imageStream);
                fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(contentType);
                content.Add(fileContent, "file", fileName);

                var response = await _httpClient.PostAsync(endpoint, content, ct);

                if (response.StatusCode == System.Net.HttpStatusCode.ServiceUnavailable)
                {
                    var rawUnavailable = await response.Content.ReadAsStringAsync(ct);
                    _logger.LogWarning(
                        "Verification API: {Endpoint} reported OCR unavailable (HTTP 503). Body: {Body}",
                        endpoint,
                        SanitizeBody(rawUnavailable));
                    try
                    {
                        return JsonSerializer.Deserialize<ChequeOcrResponseDto>(rawUnavailable, JsonDefaults.Options);
                    }
                    catch (JsonException ex)
                    {
                        _logger.LogError(ex, "Verification API: 503 deserialization failed for {Endpoint}.", endpoint);
                        return null;
                    }
                }

                return await HandleResponseAsync<ChequeOcrResponseDto>(endpoint, response, ct);
            },
            cancellationToken);
    }

    // AI V2 comparison. Additive: the OpenCV baseline above is untouched and
    // the two scores are never combined. A controlled HTTP 503 ("AI disabled,
    // checkpoint invalid or PyTorch absent") still carries a structured body
    // with success=false; it is surfaced as-is instead of being flattened to
    // null so the caller can display the server-provided reason.
    public async Task<SignatureAiComparisonResponseDto?> CompareSignaturesAiAsync(Stream extractedSignature, string extractedFileName, string extractedContentType, Stream referenceSignature, string referenceFileName, string referenceContentType, CancellationToken cancellationToken = default)
    {
        const string endpoint = "/api/signatures/compare-ai";
        return await CallAsync(
            endpoint,
            async ct =>
            {
                using var content = new MultipartFormDataContent();
                using var extractedContent = new StreamContent(extractedSignature);
                extractedContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(extractedContentType);
                content.Add(extractedContent, "extracted_file", extractedFileName);

                using var referenceContent = new StreamContent(referenceSignature);
                referenceContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(referenceContentType);
                content.Add(referenceContent, "reference_file", referenceFileName);

                var response = await _httpClient.PostAsync(endpoint, content, ct);

                if (response.StatusCode == System.Net.HttpStatusCode.ServiceUnavailable)
                {
                    var rawUnavailable = await response.Content.ReadAsStringAsync(ct);
                    _logger.LogWarning(
                        "Verification API: {Endpoint} reported AI unavailable (HTTP 503). Body: {Body}",
                        endpoint,
                        SanitizeBody(rawUnavailable));
                    try
                    {
                        return JsonSerializer.Deserialize<SignatureAiComparisonResponseDto>(rawUnavailable, JsonDefaults.Options);
                    }
                    catch (JsonException ex)
                    {
                        _logger.LogError(ex, "Verification API: 503 deserialization failed for {Endpoint}.", endpoint);
                        return null;
                    }
                }

                return await HandleResponseAsync<SignatureAiComparisonResponseDto>(endpoint, response, ct);
            },
            cancellationToken);
    }

    private async Task<TDto?> CallAsync<TDto>(string endpoint, Func<CancellationToken, Task<TDto?>> send, CancellationToken cancellationToken)
        where TDto : class
    {
        _logger.LogInformation("Verification API: calling {Endpoint}.", endpoint);
        try
        {
            return await send(cancellationToken);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            _logger.LogWarning(ex, "Verification API unreachable on {Endpoint}.", endpoint);
            return null;
        }
        finally
        {
            _logger.LogInformation("Verification API: call to {Endpoint} completed.", endpoint);
        }
    }

    private async Task<TDto?> HandleResponseAsync<TDto>(string endpoint, HttpResponseMessage response, CancellationToken cancellationToken)
        where TDto : class
    {
        _logger.LogInformation("Verification API: {Endpoint} returned HTTP {StatusCode} ({Status}).", endpoint, (int)response.StatusCode, response.StatusCode);

        if (!response.IsSuccessStatusCode)
        {
            var body = await ReadBodySafelyAsync(response, cancellationToken);
            _logger.LogWarning(
                "Verification API: {Endpoint} failed with HTTP {Status}. Body: {Body}",
                endpoint,
                response.StatusCode,
                body);
            return null;
        }

        var raw = await response.Content.ReadAsStringAsync(cancellationToken);
        _logger.LogDebug(
            "Verification API: {Endpoint} success, body length {Length} bytes (base64 fields truncated).",
            endpoint,
            raw.Length);

        try
        {
            return JsonSerializer.Deserialize<TDto>(raw, JsonDefaults.Options);
        }
        catch (JsonException ex)
        {
            _logger.LogError(
                ex,
                "Verification API: deserialization failed for {Endpoint}. Body (sanitized): {Body}",
                endpoint,
                SanitizeBody(raw));
            return null;
        }
    }

    private static async Task<string> ReadBodySafelyAsync(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        try
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken);
            return SanitizeBody(body);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or ObjectDisposedException)
        {
            return "<impossible de lire le corps>";
        }
    }

    // Internal so tests can verify no base64 image payload ever reaches the logs.
    internal static string SanitizeBody(string body)
    {
        if (string.IsNullOrEmpty(body))
        {
            return body;
        }

        // Replace full base64 image values (potentially multi-KB) with a marker.
        // Matches `"field_name": "...."` where the value is long base64.
        foreach (var field in Base64Fields)
        {
            body = Regex.Replace(
                body,
                $@"(""{field}""\s*:\s*"")[^""]*(?="")",
                m => $"{m.Groups[1].Value}[base64: {m.Value.Length - m.Groups[1].Value.Length - 2} chars]",
                RegexOptions.IgnoreCase);
        }

        return body.Length > 2000 ? body[..2000] + "…(tronqué)" : body;
    }

    private static class JsonDefaults
    {
        public static readonly System.Text.Json.JsonSerializerOptions Options = new(System.Text.Json.JsonSerializerDefaults.Web);
    }
}