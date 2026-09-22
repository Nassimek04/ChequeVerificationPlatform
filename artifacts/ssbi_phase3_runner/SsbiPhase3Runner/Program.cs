using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

var config = new ConfigurationBuilder()
    .SetBasePath(@"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web")
    .AddJsonFile("appsettings.json", optional:false)
    .AddJsonFile("appsettings.Development.json", optional:true)
    .Build();
var cs = config.GetConnectionString("ChequeVerificationConnection")!;
var apiUrl = config["VerificationApi:BaseUrl"] ?? "http://localhost:8000";
Console.WriteLine($"DB {cs}");
Console.WriteLine($"API {apiUrl}");

var sc = new ServiceCollection();
sc.AddLogging(b=>b.AddConsole().SetMinimumLevel(LogLevel.Information));
sc.AddDbContext<ChequeVerificationDbContext>(o=>o.UseSqlServer(cs));
sc.AddScoped<IReferenceSignatureService, ReferenceSignatureService>();
sc.AddScoped<IChequeService, ChequeService>();
sc.AddScoped<IVerificationService, VerificationService>();
sc.Configure<VerificationPolicyOptions>(config.GetSection(VerificationPolicyOptions.SectionName));
sc.AddHttpClient<IVerificationApiClient, VerificationApiClient>(c=>{c.BaseAddress=new Uri(apiUrl); c.Timeout=TimeSpan.FromSeconds(30);});
sc.AddSingleton<IWebHostEnvironment>(new FakeEnv());
var sp = sc.BuildServiceProvider();
using var scope = sp.CreateScope();
var db = scope.ServiceProvider.GetRequiredService<ChequeVerificationDbContext>();
var chequeService = scope.ServiceProvider.GetRequiredService<IChequeService>();
var verificationService = scope.ServiceProvider.GetRequiredService<IVerificationService>();
var webRoot = @"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web\wwwroot";
var basePkg = @"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase1\SSBI_controlled_test_signer7";
var admin = await db.Users.FirstAsync(u=>u.RoleId==1);
int custId=5;
Console.WriteLine($"Admin {admin.UserId} cust {custId}");

// Preflight
var cust = await db.Customers.FirstAsync(c=>c.CustomerId==custId);
var refs = await db.ReferenceSignatures.Where(r=>r.CustomerId==custId && r.IsActive).ToListAsync();
Console.WriteLine($"Preflight refs {refs.Count} cust {cust.CustomerNumber}");
var vr6 = await db.VerificationResults.FirstAsync(v=>v.VerificationId==6);
Console.WriteLine($"VR6 {vr6.SimilarityScore}");

// Samples to process - skip G02/G03 already attempted (G02 succeeded Id10, G03 failed Id11)
var genuine = new[]{"S7_GENUINE_04.png","S7_GENUINE_05.png","S7_GENUINE_06.png","S7_GENUINE_07.png","S7_GENUINE_08.png","S7_GENUINE_09.png","S7_GENUINE_10.png"};
var forged = new[]{"S7_FORGED_01.png","S7_FORGED_02.png","S7_FORGED_03.png","S7_FORGED_04.png","S7_FORGED_05.png","S7_FORGED_06.png","S7_FORGED_07.png","S7_FORGED_08.png","S7_FORGED_09.png","S7_FORGED_10.png"};

async Task Process(string fname, string subfolder){
    var path = Path.Combine(basePkg, subfolder, fname);
    Console.WriteLine($"\n--- {fname} ---");
    try{
        var num = await chequeService.GenerateChequeNumberAsync();
        await using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        var ff = new FormFile(fs,0,fs.Length,"ImageFile",fname){Headers=new HeaderDictionary(), ContentType="image/png"};
        var vm = new ChequeVerification.Web.ViewModels.Cheques.ChequeCreateViewModel{CustomerId=custId, ChequeNumber=num, Amount=1000m, IssueDate=DateOnly.FromDateTime(DateTime.Today), ImageFile=ff};
        var cheque = await chequeService.CreateAsync(vm, admin.UserId, webRoot);
        Console.WriteLine($"Created cheque {cheque.ChequeNumber} Id {cheque.ChequeId}");
        var ext = await verificationService.ExtractAndPersistSignatureAsync(cheque.ChequeId, admin.UserId);
        Console.WriteLine($"Extract success {ext.Success} quality {ext.ExtractionQuality} msg {ext.Message}");
        if(!ext.Success) { db.ChangeTracker.Clear(); return; }
        var ai = await verificationService.CompareAiSignaturesWithReferencesAsync(cheque.ChequeId);
        Console.WriteLine($"AI success {ai.Success} mean {ai.MeanRawScore} compared {ai.ComparedReferenceCount}");
        foreach(var c in ai.Comparisons.OrderBy(c=>c.ReferenceSignatureId)) Console.WriteLine($"  Ref {c.ReferenceSignatureId} score {c.Score}");
        try{
            var launch = await verificationService.LaunchVerificationAsync(cheque.ChequeId, admin.UserId);
            Console.WriteLine($"Launch success {launch.Success} VR {launch.VerificationId} msg {launch.Message}");
        }catch(Microsoft.EntityFrameworkCore.DbUpdateException ex){
            Console.WriteLine($"Launch DbUpdateException (likely CHECK constraint negative score): {ex.InnerException?.Message}");
            db.ChangeTracker.Clear();
        }
        db.ChangeTracker.Clear();
    }catch(Exception ex){
        Console.WriteLine($"Process exception for {fname}: {ex.GetType().Name} {ex.Message}");
        if(ex.InnerException!=null) Console.WriteLine($" Inner: {ex.InnerException.Message}");
        db.ChangeTracker.Clear();
    }
}

foreach(var f in genuine) await Process(f,"genuine_cheques");
foreach(var f in forged) await Process(f,"forged_cheques");

Console.WriteLine("\n=== DONE ===");
class FakeEnv : IWebHostEnvironment{ public string EnvironmentName{get=>"Development";set{}} public string ApplicationName{get=>"ChequeVerification.Web";set{}} public string WebRootPath{get=>@"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web\wwwroot";set{}} public string ContentRootPath{get=>@"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web";set{}} public Microsoft.Extensions.FileProviders.IFileProvider WebRootFileProvider{get=>throw new NotImplementedException(); set{}} public Microsoft.Extensions.FileProviders.IFileProvider ContentRootFileProvider{get=>throw new NotImplementedException(); set{}} }
