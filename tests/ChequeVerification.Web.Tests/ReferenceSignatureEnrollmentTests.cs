using System.Security.Cryptography;
using ChequeVerification.Web.Controllers;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace ChequeVerification.Web.Tests;

public class ReferenceSignatureEnrollmentTests : IDisposable
{
    private const string UploadsSubFolder = "uploads/signatures/reference";

    private readonly string _webRoot;

    public ReferenceSignatureEnrollmentTests()
    {
        _webRoot = Path.Combine(Path.GetTempPath(), "ref-enroll-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_webRoot);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_webRoot))
                Directory.Delete(_webRoot, recursive: true);
        }
        catch { }
    }

    private string[] ReferenceFiles()
    {
        var dir = Path.Combine(_webRoot, UploadsSubFolder);
        return Directory.Exists(dir) ? Directory.GetFiles(dir, "*.*") : Array.Empty<string>();
    }

    // ---- helpers ----

    private static readonly byte[] TinyPngBytes = Convert.FromBase64String("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==");
    // 1x1 JPEG minimal
    private static readonly byte[] TinyJpegBytes = Convert.FromBase64String("/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k=");

    private static IFormFile CreateFormFile(byte[] bytes, string fileName, string contentType)
    {
        var stream = new MemoryStream(bytes);
        return new FormFile(stream, 0, bytes.Length, "ImageFile", fileName)
        {
            Headers = new HeaderDictionary(),
            ContentType = contentType
        };
    }

    private DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"ref-enroll-db-{Guid.NewGuid():N}")
            .Options;

    private ChequeVerificationDbContext CreateDbContext() => new(CreateOptions());

    private ReferenceSignatureService CreateService(ChequeVerificationDbContext db)
        => new(db, NullLogger<ReferenceSignatureService>.Instance);

    private async Task<int> SeedCustomerAsync(ChequeVerificationDbContext db, string number = "CLI-0001")
    {
        var customer = new Customer
        {
            CustomerNumber = number,
            FullName = "Client Test " + number,
            AccountNumber = "ACC-" + number
        };
        db.Customers.Add(customer);
        await db.SaveChangesAsync();
        return customer.CustomerId;
    }

    // ---- 1 & 2: valid PNG / JPEG enrollment ----

    [Fact]
    public async Task AdministratorCanEnrollValidPng()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "signature.png", "image/png");

        var result = await service.CreateAsync(customerId, file, userId: 1, _webRoot);

        Assert.True(result.Success, result.Message);
        Assert.NotNull(result.ReferenceSignatureId);
        Assert.StartsWith($"/{UploadsSubFolder}/", result.ImagePath);
        Assert.Single(ReferenceFiles());
        var saved = await db.ReferenceSignatures.SingleAsync(r => r.CustomerId == customerId);
        Assert.Equal(result.ImagePath, saved.ImagePath);
        Assert.NotNull(saved.FileHash);
        Assert.True(saved.IsActive);
    }

    [Fact]
    public async Task AdministratorCanEnrollValidJpeg()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyJpegBytes, "signature.jpg", "image/jpeg");

        var result = await service.CreateAsync(customerId, file, userId: 1, _webRoot);

        Assert.True(result.Success, result.Message);
        Assert.Single(ReferenceFiles());
        var saved = await db.ReferenceSignatures.SingleAsync();
        Assert.EndsWith(".jpg", saved.ImagePath, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AdministratorCanEnrollValidJpeg_WithJpegExtension()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyJpegBytes, "signature.jpeg", "image/jpeg");

        var result = await service.CreateAsync(customerId, file, userId: 1, _webRoot);

        Assert.True(result.Success, result.Message);
        Assert.Single(ReferenceFiles());
    }

    // ---- 3: non-Administrator cannot enroll (controller auth) ----

    [Fact]
    public void NonAdministratorCannotEnroll_ControllerRequiresAdministrateur()
    {
        var type = typeof(ReferenceSignaturesController);

        var getMethod = type.GetMethod("Create", new[] { typeof(int), typeof(CancellationToken) });
        Assert.NotNull(getMethod);
        var getAuth = getMethod!.GetCustomAttributes(typeof(AuthorizeAttribute), false)
            .Cast<AuthorizeAttribute>().FirstOrDefault();
        Assert.NotNull(getAuth);
        Assert.Equal("Administrateur", getAuth!.Roles);

        var postMethod = type.GetMethods()
            .FirstOrDefault(m => m.Name == "Create" && m.GetParameters().Any(p => p.ParameterType.Name.Contains("CreateViewModel")));
        Assert.NotNull(postMethod);
        var postAuth = postMethod!.GetCustomAttributes(typeof(AuthorizeAttribute), false)
            .Cast<AuthorizeAttribute>().FirstOrDefault();
        Assert.NotNull(postAuth);
        Assert.Equal("Administrateur", postAuth!.Roles);

        Assert.NotNull(postMethod.GetCustomAttributes(typeof(HttpPostAttribute), false).FirstOrDefault());
        Assert.NotNull(postMethod.GetCustomAttributes(typeof(ValidateAntiForgeryTokenAttribute), false).FirstOrDefault());
        Assert.NotNull(getMethod.GetCustomAttributes(typeof(HttpGetAttribute), false).FirstOrDefault());
    }

    // ---- 4: empty file ----

    [Fact]
    public async Task EmptyFile_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(Array.Empty<byte>(), "empty.png", "image/png");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Empty(ReferenceFiles());
        Assert.False(await db.ReferenceSignatures.AnyAsync());
    }

    // ---- 5: >5 MiB ----

    [Fact]
    public async Task Over5MiB_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        // 5 MiB + 1 byte, with PNG header
        var big = new byte[5 * 1024 * 1024 + 1];
        big[0] = 0x89; big[1] = 0x50; big[2] = 0x4E; big[3] = 0x47;
        big[4] = 0x0D; big[5] = 0x0A; big[6] = 0x1A; big[7] = 0x0A;
        var file = CreateFormFile(big, "big.png", "image/png");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Contains("5 Mo", result.Message);
        Assert.Empty(ReferenceFiles());
    }

    // ---- 6: unsupported extension ----

    [Fact]
    public async Task UnsupportedExtension_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "signature.gif", "image/gif");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Contains("Format d'image non autorisé", result.Message);
        Assert.Empty(ReferenceFiles());
    }

    [Fact]
    public async Task TxtExtension_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "signature.txt", "text/plain");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Empty(ReferenceFiles());
    }

    // ---- 7: spoofed extension / invalid magic bytes ----

    [Fact]
    public async Task SpoofedExtension_InvalidMagicBytes_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var textBytes = System.Text.Encoding.UTF8.GetBytes("This is not an image");
        var file = CreateFormFile(textBytes, "fake.png", "image/png");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Contains("ne correspond pas", result.Message);
        Assert.Empty(ReferenceFiles());
    }

    [Fact]
    public async Task JpegExtensionWithPngContent_SucceedsIfMagicMatches()
    {
        // PNG bytes with .jpg extension but correct magic -> extension valid, magic valid => should succeed (extension .jpg is allowed, magic PNG is valid)
        // Our HasValidMagicBytes accepts PNG or JPEG regardless of extension.
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "photo.jpg", "image/jpeg");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        // Content is PNG but extension .jpg and contentType jpeg -> magic passes (PNG magic is valid)
        // So this should succeed because magic is valid PNG.
        Assert.True(result.Success, result.Message);
    }

    [Fact]
    public async Task RandomBytesWithValidExtension_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var random = new byte[100];
        new Random(42).NextBytes(random);
        // Ensure not PNG/JPEG magic
        random[0] = 0x00; random[1] = 0x00; random[2] = 0x00;
        var file = CreateFormFile(random, "random.png", "image/png");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Empty(ReferenceFiles());
    }

    // ---- 8: invalid ContentType ----

    [Fact]
    public async Task InvalidContentType_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "signature.png", "text/plain");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Contains("Type de contenu", result.Message);
        Assert.Empty(ReferenceFiles());
    }

    [Fact]
    public async Task ApplicationOctetStream_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "signature.png", "application/octet-stream");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Empty(ReferenceFiles());
    }

    // ---- 9: duplicate SHA-256 same customer ----

    [Fact]
    public async Task DuplicateSha256_SameCustomer_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);

        var file1 = CreateFormFile(TinyPngBytes, "sig1.png", "image/png");
        var r1 = await service.CreateAsync(customerId, file1, 1, _webRoot);
        Assert.True(r1.Success);

        var file2 = CreateFormFile(TinyPngBytes, "sig2.png", "image/png");
        var r2 = await service.CreateAsync(customerId, file2, 1, _webRoot);

        Assert.False(r2.Success);
        Assert.Contains("identique existe déjà", r2.Message);
        Assert.Single(await db.ReferenceSignatures.ToListAsync());
        Assert.Single(ReferenceFiles()); // no orphan
    }

    // ---- 10: same image different customer -> allowed ----

    [Fact]
    public async Task SameImage_DifferentCustomer_Allowed()
    {
        await using var db = CreateDbContext();
        var c1 = await SeedCustomerAsync(db, "CLI-0001");
        var c2 = await SeedCustomerAsync(db, "CLI-0002");
        var service = CreateService(db);

        var f1 = CreateFormFile(TinyPngBytes, "sig.png", "image/png");
        var r1 = await service.CreateAsync(c1, f1, 1, _webRoot);
        Assert.True(r1.Success);

        var f2 = CreateFormFile(TinyPngBytes, "sig.png", "image/png");
        var r2 = await service.CreateAsync(c2, f2, 1, _webRoot);
        Assert.True(r2.Success, r2.Message);

        Assert.Equal(2, await db.ReferenceSignatures.CountAsync());
        Assert.Equal(2, ReferenceFiles().Length);
    }

    // ---- 11: sixth active rejected ----

    [Fact]
    public async Task SixthActiveReference_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);

        for (int i = 0; i < 5; i++)
        {
            var variant = new byte[TinyPngBytes.Length];
            TinyPngBytes.CopyTo(variant, 0);
            variant[^1] ^= (byte)(i + 1); // distinct bytes -> distinct hash but valid PNG
            // Keep PNG magic intact, tweak last byte only
            var f = CreateFormFile(variant, $"sig{i}.png", "image/png");
            var r = await service.CreateAsync(customerId, f, 1, _webRoot);
            Assert.True(r.Success, $"Failed at i={i}: {r.Message}");
        }

        Assert.Equal(5, await db.ReferenceSignatures.CountAsync(r => r.CustomerId == customerId && r.IsActive));
        Assert.Equal(5, ReferenceFiles().Length);

        var extra = new byte[TinyPngBytes.Length];
        TinyPngBytes.CopyTo(extra, 0);
        extra[^1] ^= 0xFF;
        var file6 = CreateFormFile(extra, "sig6.png", "image/png");
        var result = await service.CreateAsync(customerId, file6, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Contains("déjà 5 signatures", result.Message);
        Assert.Equal(5, await db.ReferenceSignatures.CountAsync());
        Assert.Equal(5, ReferenceFiles().Length);
    }

    // ---- 12: generated filename not original ----

    [Fact]
    public async Task GeneratedFilename_NotOriginal()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "my-secret-signature.png", "image/png");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.True(result.Success);
        Assert.NotNull(result.ImagePath);
        Assert.DoesNotContain("my-secret-signature", result.ImagePath);
        Assert.StartsWith($"/{UploadsSubFolder}/", result.ImagePath);
        var fileName = Path.GetFileName(result.ImagePath);
        Assert.True(Guid.TryParse(Path.GetFileNameWithoutExtension(fileName), out _), "Filename should be GUID");
        Assert.EndsWith(".png", fileName, StringComparison.OrdinalIgnoreCase);
        var physical = Path.Combine(_webRoot, result.ImagePath.TrimStart('/').Replace('/', Path.DirectorySeparatorChar));
        Assert.True(File.Exists(physical));
        Assert.Single(ReferenceFiles());
    }

    // ---- 13: DB failure cleans file ----

    [Fact]
    public async Task DbFailure_CleansNewFile()
    {
        await using var db = new FailingSaveChangesDbContext(CreateOptions());
        var customerId = await SeedCustomerAsync(db);
        db.FailNextSave = true;
        var service = new ReferenceSignatureService(db, NullLogger<ReferenceSignatureService>.Instance);
        var file = CreateFormFile(TinyPngBytes, "sig.png", "image/png");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Empty(ReferenceFiles());
        Assert.False(await db.ReferenceSignatures.AnyAsync());
    }

    // ---- 14: audit log ----

    [Fact]
    public async Task SuccessfulEnrollment_CreatesAuditLog()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "sig.png", "image/png");

        var result = await service.CreateAsync(customerId, file, userId: 42, _webRoot);

        Assert.True(result.Success);
        var audit = await db.AuditLogs.SingleAsync();
        Assert.Equal("ENROLL_REFERENCE", audit.Action);
        Assert.Equal(nameof(ReferenceSignature), audit.EntityName);
        Assert.Equal(result.ReferenceSignatureId, audit.EntityId);
        Assert.Equal(42, audit.UserId);
        Assert.Contains(customerId.ToString(), audit.Description);
    }

    // ---- 15: FileHash ----

    [Fact]
    public async Task SuccessfulEnrollment_StoresFileHash()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "sig.png", "image/png");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.True(result.Success);
        var expected = Convert.ToHexString(SHA256.HashData(TinyPngBytes)).ToLowerInvariant();
        Assert.Equal(expected, result.FileHash);
        var saved = await db.ReferenceSignatures.SingleAsync();
        Assert.Equal(expected, saved.FileHash);
    }

    // ---- 16: no existing reference modified ----

    [Fact]
    public async Task NoExistingReferenceModified_OnNewEnrollment()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);

        var f1 = CreateFormFile(TinyPngBytes, "sig1.png", "image/png");
        var r1 = await service.CreateAsync(customerId, f1, 1, _webRoot);
        Assert.True(r1.Success);
        var first = await db.ReferenceSignatures.SingleAsync();

        var variant = new byte[TinyPngBytes.Length];
        TinyPngBytes.CopyTo(variant, 0);
        variant[^1] ^= 0x01;
        var f2 = CreateFormFile(variant, "sig2.png", "image/png");
        var r2 = await service.CreateAsync(customerId, f2, 1, _webRoot);
        Assert.True(r2.Success);

        var all = await db.ReferenceSignatures.OrderBy(r => r.ReferenceSignatureId).ToListAsync();
        Assert.Equal(2, all.Count);
        var stillFirst = all.First(r => r.ReferenceSignatureId == first.ReferenceSignatureId);
        Assert.Equal(first.ImagePath, stillFirst.ImagePath);
        Assert.Equal(first.FileHash, stillFirst.FileHash);
        Assert.Equal(first.CreatedAt, stillFirst.CreatedAt);
        Assert.Equal(2, ReferenceFiles().Length);
    }

    // ---- additional edge cases ----

    [Fact]
    public async Task NullFile_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);

        var result = await service.CreateAsync(customerId, null!, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Contains("Aucun fichier", result.Message);
        Assert.Empty(ReferenceFiles());
    }

    [Fact]
    public async Task CustomerNotFound_Rejected()
    {
        await using var db = CreateDbContext();
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "sig.png", "image/png");

        var result = await service.CreateAsync(customerId: 9999, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Contains("introuvable", result.Message);
        Assert.Empty(ReferenceFiles());
    }

    [Fact]
    public async Task PathTraversalInFilename_Rejected()
    {
        await using var db = CreateDbContext();
        var customerId = await SeedCustomerAsync(db);
        var service = CreateService(db);
        var file = CreateFormFile(TinyPngBytes, "../evil.png", "image/png");

        var result = await service.CreateAsync(customerId, file, 1, _webRoot);

        Assert.False(result.Success);
        Assert.Contains("Nom de fichier invalide", result.Message);
        Assert.Empty(ReferenceFiles());
    }

    // ---- helpers for failing db ----

    private sealed class FailingSaveChangesDbContext : ChequeVerificationDbContext
    {
        public FailingSaveChangesDbContext(DbContextOptions<ChequeVerificationDbContext> options) : base(options) { }
        public bool FailNextSave { get; set; }
        public override Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
        {
            if (FailNextSave)
            {
                FailNextSave = false;
                throw new DbUpdateException("Erreur SQL simulée.");
            }
            return base.SaveChangesAsync(cancellationToken);
        }
    }
}
