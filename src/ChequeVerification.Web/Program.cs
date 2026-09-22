using ChequeVerification.Web.Data;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllersWithViews();

builder.Services.AddDbContext<ChequeVerificationDbContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("ChequeVerificationConnection")));

builder.Services.AddScoped<IAuthService, AuthService>();
builder.Services.AddScoped<IDashboardService, DashboardService>();
builder.Services.AddScoped<ICustomerService, CustomerService>();
builder.Services.AddScoped<IReferenceSignatureService, ReferenceSignatureService>();
builder.Services.AddScoped<IChequeService, ChequeService>();
builder.Services.AddScoped<IChequeBatchImportService, ChequeBatchImportService>();
builder.Services.Configure<VerificationPolicyOptions>(builder.Configuration.GetSection(VerificationPolicyOptions.SectionName));
builder.Services.AddScoped<IVerificationService, VerificationService>();
builder.Services.AddScoped<IManualReviewService, ManualReviewService>();
builder.Services.AddScoped<IPasswordHasher<ChequeVerification.Web.Models.Entities.User>, PasswordHasher<ChequeVerification.Web.Models.Entities.User>>();

builder.Services.AddHttpClient<IVerificationApiClient, VerificationApiClient>(client =>
{
    client.BaseAddress = new Uri(builder.Configuration["VerificationApi:BaseUrl"] ?? "http://localhost:8000");
    // OCR cold-start + PaddleOCR inference can take ~3-5s warm and up to ~60s cold (ocr_worker_timeout).
    // The previous 5s timeout caused TaskCanceledException for the real OCR path while /health stayed fast.
    client.Timeout = TimeSpan.FromSeconds(70);
});

builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
    .AddCookie(options =>
    {
        options.LoginPath = "/Account/Login";
        options.AccessDeniedPath = "/Account/AccessDenied";
        options.ExpireTimeSpan = TimeSpan.FromMinutes(60);
        options.SlidingExpiration = true;
        options.Cookie.HttpOnly = true;
        options.Cookie.SameSite = SameSiteMode.Lax;
        options.Cookie.SecurePolicy = builder.Environment.IsDevelopment()
            ? CookieSecurePolicy.SameAsRequest
            : CookieSecurePolicy.Always;
    });

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
}

app.UseStaticFiles();

app.UseRouting();

app.UseAuthentication();
app.UseAuthorization();

app.MapStaticAssets();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}")
    .WithStaticAssets();

app.Run();
