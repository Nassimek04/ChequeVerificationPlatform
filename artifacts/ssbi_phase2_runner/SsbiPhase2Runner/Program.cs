using System.Security.Cryptography;
using System.Text;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

Console.WriteLine("=== SSBI Phase 2 Runner ===");

// Configuration
var config = new ConfigurationBuilder()
    .SetBasePath(@"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web")
    .AddJsonFile("appsettings.json", optional: false)
    .AddJsonFile("appsettings.Development.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

var connectionString = config.GetConnectionString("ChequeVerificationConnection") ?? "Server=localhost\\SQLEXPRESS;Database=ChequeVerificationDB;Trusted_Connection=True;TrustServerCertificate=True";
Console.WriteLine($"Connection: {connectionString}");

var verificationApiBaseUrl = config["VerificationApi:BaseUrl"] ?? "http://localhost:8000";
Console.WriteLine($"VerificationApi: {verificationApiBaseUrl}");

var services = new ServiceCollection();
services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Information));
services.AddDbContext<ChequeVerificationDbContext>(o => o.UseSqlServer(connectionString));
services.AddScoped<IReferenceSignatureService, ReferenceSignatureService>();
services.AddScoped<IChequeService, ChequeService>();
services.AddScoped<IVerificationService, VerificationService>();
services.Configure<VerificationPolicyOptions>(config.GetSection(VerificationPolicyOptions.SectionName));
services.AddHttpClient<IVerificationApiClient, VerificationApiClient>(c => { c.BaseAddress = new Uri(verificationApiBaseUrl); c.Timeout = TimeSpan.FromSeconds(30); });
services.AddSingleton<IWebHostEnvironment>(new FakeEnv());

var sp = services.BuildServiceProvider();
using var scope = sp.CreateScope();
var db = scope.ServiceProvider.GetRequiredService<ChequeVerificationDbContext>();
var loggerFactory = scope.ServiceProvider.GetRequiredService<ILoggerFactory>();
var refService = scope.ServiceProvider.GetRequiredService<IReferenceSignatureService>();
var chequeService = scope.ServiceProvider.GetRequiredService<IChequeService>();
var verificationService = scope.ServiceProvider.GetRequiredService<IVerificationService>();
var env = scope.ServiceProvider.GetRequiredService<IWebHostEnvironment>();

// Ensure DB can connect
Console.WriteLine($"DB CanConnect: {await db.Database.CanConnectAsync()}");

// Step 1: Customer preflight
var existing = await db.Customers.AsNoTracking().FirstOrDefaultAsync(c => c.CustomerNumber == "CLI-SSBI-0007" || c.FullName == "SSBI Test Signer 7");
if (existing != null)
{
    Console.WriteLine($"[STEP1] SSBI customer already exists: Id={existing.CustomerId} Number={existing.CustomerNumber} Name={existing.FullName} Account={existing.AccountNumber}");
}
else
{
    Console.WriteLine("[STEP1] SSBI Test Signer 7 NOT found - creating via EF Core (secure, not raw SQL string)");
    // Create admin user if needed
    var admin = await db.Users.FirstOrDefaultAsync(u => u.Email == "admin.ssbi@demo.local");
    if (admin == null)
    {
        var hasher = new Microsoft.AspNetCore.Identity.PasswordHasher<User>();
        var adminUser = new User { FullName = "SSBI Admin", Email = "admin.ssbi@demo.local", RoleId = 1, Status = 1, CreatedAt = DateTime.UtcNow };
        adminUser.PasswordHash = hasher.HashPassword(adminUser, "Admin123!");
        db.Users.Add(adminUser);
        await db.SaveChangesAsync();
        Console.WriteLine($"Created Administrateur user Id={adminUser.UserId} Email={adminUser.Email}");
        admin = adminUser;
    }
    else Console.WriteLine($"Administrateur exists Id={admin.UserId}");

    var customer = new Customer
    {
        CustomerNumber = "CLI-SSBI-0007",
        FullName = "SSBI Test Signer 7",
        AccountNumber = "ACC-SSBI-0007",
        NationalId = "SSBI-7",
        Phone = null,
        Email = null,
        CreatedAt = DateTime.UtcNow
    };
    db.Customers.Add(customer);
    await db.SaveChangesAsync();
    Console.WriteLine($"[STEP1] Created customer Id={customer.CustomerId} Number={customer.CustomerNumber}");

    // Audit log for customer creation (best-effort)
    db.AuditLogs.Add(new AuditLog { UserId = admin.UserId, Action = "CREATE_CUSTOMER", EntityName = nameof(Customer), EntityId = customer.CustomerId, Description = $"Création du client SSBI Test Signer 7 ({customer.CustomerNumber})", CreatedAt = DateTime.UtcNow });
    await db.SaveChangesAsync();

    existing = await db.Customers.AsNoTracking().FirstAsync(c => c.CustomerId == customer.CustomerId);
}
int ssbiCustomerId = existing.CustomerId;
Console.WriteLine($"[STEP1] Using SSBI CustomerId={ssbiCustomerId}");

// Verify CEDAR not touched
var cedar = await db.Customers.Where(c => c.CustomerNumber == "CLI-0003").Select(c => new { c.CustomerId, c.FullName }).FirstOrDefaultAsync();
Console.WriteLine($"[CHECK] CEDAR Client Test 7: Id={cedar?.CustomerId} Name={cedar?.FullName} (must be untouched)");

// Step 2: Reference enrollment
var activeRefs = await db.ReferenceSignatures.Where(r => r.CustomerId == ssbiCustomerId && r.IsActive).OrderBy(r => r.ReferenceSignatureId).ToListAsync();
Console.WriteLine($"[STEP2] Active references before: {activeRefs.Count}");
var webRoot = @"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web\wwwroot";
var refBase = @"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase1\SSBI_controlled_test_signer7\references";
var refFiles = new[] { "REF_S7_01.png", "REF_S7_02.png", "REF_S7_03.png", "REF_S7_04.png", "REF_S7_05.png" };

// If less than 5, enroll missing via service
var adminUserForRef = await db.Users.FirstOrDefaultAsync(u => u.RoleId == 1);
int? adminId = adminUserForRef?.UserId;
Console.WriteLine($"Using adminId for enrollment: {adminId} ({adminUserForRef?.Email})");

foreach (var f in refFiles)
{
    var existsRefByHash = false;
    var path = Path.Combine(refBase, f);
    var bytes = await File.ReadAllBytesAsync(path);
    var hash = ComputeSha256Hex(bytes);
    var dup = await db.ReferenceSignatures.AnyAsync(r => r.CustomerId == ssbiCustomerId && r.FileHash == hash);
    if (dup) { Console.WriteLine($"  {f} already enrolled (hash {hash[..8]}...) skip"); continue; }

    // Enroll via service
    await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
    var formFile = new FormFile(stream, 0, stream.Length, "file", f)
    {
        Headers = new HeaderDictionary(),
        ContentType = f.EndsWith(".png") ? "image/png" : "image/jpeg"
    };
    var result = await refService.CreateAsync(ssbiCustomerId, formFile, adminId, webRoot);
    Console.WriteLine($"  Enroll {f}: Success={result.Success} Message={result.Message} Id={result.ReferenceSignatureId} Hash={result.FileHash?[..8]}");
    if (!result.Success) Console.WriteLine($"    FAILED to enroll {f}");
}

activeRefs = await db.ReferenceSignatures.Where(r => r.CustomerId == ssbiCustomerId && r.IsActive).OrderBy(r => r.ReferenceSignatureId).ToListAsync();
Console.WriteLine($"[STEP2] Active references after: {activeRefs.Count}");
foreach (var r in activeRefs) Console.WriteLine($"  REF Id={r.ReferenceSignatureId} Path={r.ImagePath} Hash={r.FileHash[..16]}... IsActive={r.IsActive}");

// Verify exactly 5
if (activeRefs.Count != 5) Console.WriteLine($"[STEP2] ERROR: expected 5 active refs, got {activeRefs.Count}");

// Ensure none belong to CEDAR (check hashes not matching CEDAR refs)
var cedarRefs = await db.ReferenceSignatures.Where(r => r.CustomerId == 4).Select(r => r.FileHash).ToListAsync();
foreach (var r in activeRefs) if (cedarRefs.Contains(r.FileHash)) Console.WriteLine($"  WARNING: ref {r.ReferenceSignatureId} hash matches CEDAR!");

// Step 3: Cheque creation
var chequeNumber = await chequeService.GenerateChequeNumberAsync();
Console.WriteLine($"[STEP3] Next cheque number: {chequeNumber}");
var genuinePath = @"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase1\SSBI_controlled_test_signer7\genuine_cheques\S7_GENUINE_01.png";
var existingCheque = await db.Cheques.FirstOrDefaultAsync(c => c.CustomerId == ssbiCustomerId && c.ImagePath.Contains("S7_GENUINE_01"));
if (existingCheque != null) Console.WriteLine($"[STEP3] Cheque for S7_GENUINE_01 already exists: Id={existingCheque.ChequeId} Number={existingCheque.ChequeNumber} Status={existingCheque.Status}");
else
{
    // Create cheque via service
    var importer = adminUserForRef ?? await db.Users.FirstAsync();
    await using var imgStream = new FileStream(genuinePath, FileMode.Open, FileAccess.Read);
    var imgFormFile = new FormFile(imgStream, 0, imgStream.Length, "ImageFile", "S7_GENUINE_01.png")
    {
        Headers = new HeaderDictionary(),
        ContentType = "image/png"
    };
    var vm = new ChequeVerification.Web.ViewModels.Cheques.ChequeCreateViewModel
    {
        CustomerId = ssbiCustomerId,
        ChequeNumber = chequeNumber,
        Amount = 1000.00m,
        IssueDate = DateOnly.FromDateTime(DateTime.Today),
        ImageFile = imgFormFile
    };
    var cheque = await chequeService.CreateAsync(vm, importer.UserId, webRoot);
    Console.WriteLine($"[STEP3] Created cheque Id={cheque.ChequeId} Number={cheque.ChequeNumber} Path={cheque.ImagePath} Status={cheque.Status}");
    existingCheque = cheque;
}
int chequeId = existingCheque.ChequeId;
Console.WriteLine($"[STEP3] Using ChequeId={chequeId}");

// Step 4: Extraction
Console.WriteLine($"[STEP4] Extracting signature for cheque {chequeId}");
var extractResult = await verificationService.ExtractAndPersistSignatureAsync(chequeId, adminUserForRef?.UserId);
Console.WriteLine($"[STEP4] Extract Success={extractResult.Success} Message={extractResult.Message} ExtractedId={extractResult.ExtractedSignatureId} Path={extractResult.ImagePath} Quality={extractResult.ExtractionQuality}");

if (!extractResult.Success)
{
    var prep = await verificationService.PrepareVerificationAsync(chequeId);
    Console.WriteLine($"  Preparation blocking: {prep?.BlockingReason}");
    // Try to get more detail via direct service eligibility
}

// Fetch extracted signature
var extracted = await db.ExtractedSignatures.AsNoTracking().FirstOrDefaultAsync(e => e.ChequeId == chequeId);
if (extracted != null)
{
    Console.WriteLine($"[STEP4] ExtractedSignature: Id={extracted.ExtractedSignatureId} Path={extracted.ImagePath} Hash={extracted.FileHash[..16]}... Confidence={extracted.ExtractionConfidence} At={extracted.ExtractedAt}");
    var phys = Path.Combine(webRoot, extracted.ImagePath.TrimStart('/','\\').Replace('/', Path.DirectorySeparatorChar));
    Console.WriteLine($"  Physical exists: {File.Exists(phys)}");
    if (File.Exists(phys))
    {
        var info = new FileInfo(phys);
        Console.WriteLine($"  File size: {info.Length} bytes");
    }
    // Compare with Phase1 expected bbox via API debug? We'll just report quality
    Console.WriteLine($"  Expected approx bbox 1562,478,253,101 quality 0.7852 completeness 1.0");
}

// Step 5: AI V2 K=5 comparison
Console.WriteLine($"[STEP5] AI V2 comparison for cheque {chequeId}");
var aiResult = await verificationService.CompareAiSignaturesWithReferencesAsync(chequeId);
Console.WriteLine($"[STEP5] Success={aiResult.Success} Message={aiResult.Message}");
Console.WriteLine($"  ActiveReferenceCount={aiResult.ActiveReferenceCount} Compared={aiResult.ComparedReferenceCount} Unavailable={aiResult.UnavailableReferenceCount} IsAggregation={aiResult.IsAggregationAvailable} MeanRaw={aiResult.MeanRawScore}");
if (aiResult.Comparisons != null)
{
    foreach (var c in aiResult.Comparisons.OrderBy(c => c.ReferenceSignatureId))
    {
        Console.WriteLine($"  REF Id={c.ReferenceSignatureId} Available={c.IsAvailable} Score={c.Score} Method={c.Method} Ver={c.Version} Msg={c.StatusMessage}");
    }
}
double? mean = aiResult.MeanRawScore;
if (mean.HasValue) Console.WriteLine($"  MeanRawScore = {mean.Value:F6}");

// Step 6: Decision
if (mean.HasValue)
{
    var opts = scope.ServiceProvider.GetRequiredService<IOptions<VerificationPolicyOptions>>().Value;
    Console.WriteLine($"[STEP6] Policy L={opts.LowerThreshold} U={opts.UpperThreshold} Score={mean.Value:F6}");
    string decision;
    if (mean.Value <= (double)opts.LowerThreshold) decision = "Non conforme";
    else if (mean.Value >= (double)opts.UpperThreshold) decision = "Conforme";
    else decision = "Contrôle manuel";
    Console.WriteLine($"[STEP6] AutomaticDecision = {decision}");
}

// Step 7: LaunchVerification persistence
Console.WriteLine($"[STEP7] LaunchVerification for cheque {chequeId}");
var launch = await verificationService.LaunchVerificationAsync(chequeId, adminUserForRef?.UserId ?? 1);
Console.WriteLine($"[STEP7] Launch Success={launch.Success} Message={launch.Message} VerificationId={launch.VerificationId}");

if (launch.Success && launch.VerificationId.HasValue)
{
    var vr = await db.VerificationResults.AsNoTracking().Include(v => v.SignatureComparisons).FirstOrDefaultAsync(v => v.VerificationId == launch.VerificationId.Value);
    if (vr != null)
    {
        Console.WriteLine($"  VR: ChequeId={vr.ChequeId} SimilarityScore={vr.SimilarityScore} Lower={vr.LowerThresholdUsed} Upper={vr.UpperThresholdUsed} Auto={vr.AutomaticDecision} Final={vr.FinalDecision} Model={vr.ModelName} Ver={vr.ModelVersion} At={vr.VerifiedAt}");
        Console.WriteLine($"  SignatureComparisons rows: {vr.SignatureComparisons.Count}");
        foreach (var sc in vr.SignatureComparisons.OrderBy(s => s.ReferenceSignatureId))
            Console.WriteLine($"    SC CompId={sc.ComparisonId} RefId={sc.ReferenceSignatureId} Score={sc.SimilarityScore} IsBest={sc.IsBestMatch}");
        var best = vr.SignatureComparisons.FirstOrDefault(s => s.IsBestMatch);
        Console.WriteLine($"  Best RefId={best?.ReferenceSignatureId}");
    }
    var chequeAfter = await db.Cheques.AsNoTracking().FirstAsync(c => c.ChequeId == chequeId);
    Console.WriteLine($"  Cheque final Status={chequeAfter.Status} (1=EnAttente 2=EnTraitement 3=Verifie 4=ControleManuel 5=Rejete 6=Erreur)");
    var audit = await db.AuditLogs.OrderByDescending(a => a.AuditLogId).FirstOrDefaultAsync(a => a.Action == "VERIFY_CHEQUE" && a.EntityId == vr.VerificationId);
    Console.WriteLine($"  AuditLog VERIFY_CHEQUE: Id={audit?.AuditLogId} Desc={audit?.Description}");
}
else
{
    var hasVr = await db.VerificationResults.AnyAsync(v => v.ChequeId == chequeId);
    Console.WriteLine($"  Has existing VR: {hasVr}");
    if (hasVr)
    {
        var vr2 = await db.VerificationResults.AsNoTracking().Include(v => v.SignatureComparisons).FirstAsync(v => v.ChequeId == chequeId);
        Console.WriteLine($"  Existing VR Id={vr2.VerificationId} Score={vr2.SimilarityScore} Auto={vr2.AutomaticDecision}");
    }
}

Console.WriteLine("=== DONE ===");

static string ComputeSha256Hex(byte[] bytes)
{
    var hash = SHA256.HashData(bytes);
    var sb = new StringBuilder(hash.Length * 2);
    foreach (var b in hash) sb.Append(b.ToString("x2"));
    return sb.ToString();
}

class FakeEnv : IWebHostEnvironment
{
    public string EnvironmentName { get => "Development"; set => throw new NotImplementedException(); }
    public string ApplicationName { get => "ChequeVerification.Web"; set => throw new NotImplementedException(); }
    public string WebRootPath { get => @"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web\wwwroot"; set => throw new NotImplementedException(); }
    public string ContentRootPath { get => @"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web"; set => throw new NotImplementedException(); }
    public Microsoft.Extensions.FileProviders.IFileProvider WebRootFileProvider { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
    public Microsoft.Extensions.FileProviders.IFileProvider ContentRootFileProvider { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
}
