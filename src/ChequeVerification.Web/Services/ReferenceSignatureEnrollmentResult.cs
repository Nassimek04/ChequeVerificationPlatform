namespace ChequeVerification.Web.Services;

public class ReferenceSignatureEnrollmentResult
{
    public bool Success { get; set; }

    public string Message { get; set; } = string.Empty;

    public int? ReferenceSignatureId { get; set; }

    public string? ImagePath { get; set; }

    public string? FileHash { get; set; }
}
