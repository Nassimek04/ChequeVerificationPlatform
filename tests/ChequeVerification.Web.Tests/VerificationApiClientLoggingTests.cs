using ChequeVerification.Web.Services;

namespace ChequeVerification.Web.Tests;

public class VerificationApiClientLoggingTests
{
    // Every base64 image field that must never be logged in full.
    private static readonly string[] Base64ImageFields =
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
        "refined_group_image_base64"
    };

    [Fact]
    public void SanitizeBody_ReplacesEveryBase64ImageField()
    {
        var payloads = Base64ImageFields.ToDictionary(f => f, f => new string(f[0], 5000) + "payload");
        var json = "{" + string.Join(",", Base64ImageFields.Select(f => $"\"{f}\": \"{payloads[f]}\"")) + "}";

        var sanitized = VerificationApiClient.SanitizeBody(json);

        // No complete base64 payload may remain anywhere in the sanitized body.
        foreach (var field in Base64ImageFields)
        {
            Assert.DoesNotContain(payloads[field], sanitized, StringComparison.Ordinal);
            Assert.Contains("[base64:", sanitized, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SanitizeBody_ReplacesDebugImageFields_NewerV22FieldsIncluded()
    {
        foreach (var field in new[]
                 {
                     "mask_image_base64",
                     "components_all_base64",
                     "components_rejected_base64",
                     "components_image_base64",
                     "group_image_base64",
                     "groups_image_base64",
                     "refined_group_image_base64"
                 })
        {
            var json = $"{{\"{field}\": \"{new string('X', 4096)}\"}}";
            var sanitized = VerificationApiClient.SanitizeBody(json);

            Assert.DoesNotContain(new string('X', 4096), sanitized, StringComparison.Ordinal);
            Assert.Contains("[base64:", sanitized, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SanitizeBody_TruncatesVeryLongBodies()
    {
        var json = "{\"detail\": \"" + new string('D', 4000) + "\"}";
        var sanitized = VerificationApiClient.SanitizeBody(json);

        Assert.Contains("(tronqué)", sanitized, StringComparison.Ordinal);
    }

    [Fact]
    public void SanitizeBody_ShortBodyWithoutBase64_IsUnchanged()
    {
        const string json = "{\"success\": true, \"detail\": \"ok\"}";
        var sanitized = VerificationApiClient.SanitizeBody(json);

        Assert.Equal(json, sanitized);
    }
}