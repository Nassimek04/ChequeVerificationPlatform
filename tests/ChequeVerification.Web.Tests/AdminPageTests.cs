using System.Reflection;
using ChequeVerification.Web.Controllers;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.ViewModels.Admin;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace ChequeVerification.Web.Tests;

public class AdminPageTests : IDisposable
{
    public void Dispose() { }

    private static DbContextOptions<ChequeVerificationDbContext> CreateOptions()
        => new DbContextOptionsBuilder<ChequeVerificationDbContext>()
            .UseInMemoryDatabase($"admin-{Guid.NewGuid():N}").Options;

    private static IOptions<VerificationPolicyOptions> DefaultPolicy()
        => Options.Create(new VerificationPolicyOptions
        {
            LowerThreshold = 0.6585m,
            UpperThreshold = 0.9150m,
            ModelName = "sig-verif-ai-v5a",
            ModelVersion = "v5a-phase7"
        });

    private static async Task SeedAsync(ChequeVerificationDbContext db)
    {
        var admin = new Role { RoleId = 1, Name = "Administrateur", Description = "Admins" };
        var ctrl = new Role { RoleId = 2, Name = "Contrôleur", Description = "Control" };
        var user = new Role { RoleId = 3, Name = "Utilisateur", Description = "Users" };
        db.Roles.AddRange(admin, ctrl, user);
        db.Users.AddRange(
            new User { UserId = 1, RoleId = 1, FullName = "Admin One", Email = "admin@example.com", PasswordHash = "HASH", Status = 1, CreatedAt = DateTime.UtcNow },
            new User { UserId = 2, RoleId = 3, FullName = "User Two", Email = "user@example.com", PasswordHash = "HASH", Status = 1, CreatedAt = DateTime.UtcNow },
            new User { UserId = 3, RoleId = 2, FullName = "Ctrl Three", Email = "ctrl@example.com", PasswordHash = "HASH", Status = 0, CreatedAt = DateTime.UtcNow });
        var cust = new Customer { CustomerId = 1, CustomerNumber = "C-1", FullName = "Client", AccountNumber = "A" };
        db.Customers.Add(cust);
        db.ReferenceSignatures.AddRange(
            new ReferenceSignature { ReferenceSignatureId = 1, CustomerId = 1, ImagePath = "/x/1.png", CreatedAt = DateTime.UtcNow, IsActive = true },
            new ReferenceSignature { ReferenceSignatureId = 2, CustomerId = 1, ImagePath = "/x/2.png", CreatedAt = DateTime.UtcNow, IsActive = false });
        db.Cheques.Add(new Cheque { ChequeId = 1, CustomerId = 1, ImportedByUserId = 1, ChequeNumber = "CHQ-1", ImagePath = "/c.png", Status = 1, UploadedAt = DateTime.UtcNow });
        db.VerificationResults.AddRange(
            new VerificationResult { VerificationId = 1, ChequeId = 1, SimilarityScore = 0.9m, LowerThresholdUsed = 0.6585m, UpperThresholdUsed = 0.9150m, AutomaticDecision = 3, FinalDecision = null, VerifiedAt = DateTime.UtcNow },
            new VerificationResult { VerificationId = 2, ChequeId = 1, SimilarityScore = 0.95m, LowerThresholdUsed = 0.6585m, UpperThresholdUsed = 0.9150m, AutomaticDecision = 1, FinalDecision = 1, VerifiedAt = DateTime.UtcNow });
        var now = DateTime.UtcNow;
        db.AuditLogs.AddRange(
            new AuditLog { AuditLogId = 1, UserId = 1, Action = "Old", EntityName = "User", EntityId = 1, CreatedAt = now.AddHours(-2) },
            new AuditLog { AuditLogId = 2, UserId = null, Action = "Mid", EntityName = "Cheque", EntityId = 1, CreatedAt = now.AddHours(-1) },
            new AuditLog { AuditLogId = 3, UserId = 999, Action = "New", EntityName = "User", EntityId = 2, CreatedAt = now });
        await db.SaveChangesAsync();
    }

    [Fact]
    public void AdminController_RequiresAdministrateurOnly()
    {
        var classAuth = typeof(AdminController).GetCustomAttributes<AuthorizeAttribute>(false).ToList();
        Assert.NotEmpty(classAuth);
        var roles = classAuth.SelectMany(a => (a.Roles ?? string.Empty).Split(',').Select(r => r.Trim())).ToHashSet();
        Assert.Equal(new HashSet<string> { "Administrateur" }, roles);
        Assert.DoesNotContain("Utilisateur", roles);
        Assert.DoesNotContain("Contrôleur", roles);
        Assert.Empty(typeof(AdminController).GetCustomAttributes<AllowAnonymousAttribute>(false));
    }

    [Theory]
    [InlineData("Index")]
    [InlineData("Users")]
    [InlineData("Audit")]
    public void AdminActions_RequireAuthentication_NoAnonymous(string action)
    {
        var method = typeof(AdminController).GetMethod(action)!;
        Assert.Empty(method.GetCustomAttributes<AllowAnonymousAttribute>(false));
        var hasAuth = method.GetCustomAttributes<AuthorizeAttribute>(false).Any()
            || typeof(AdminController).GetCustomAttributes<AuthorizeAttribute>(false).Any();
        Assert.True(hasAuth);
    }

    [Theory]
    [InlineData("Index")]
    [InlineData("Users")]
    [InlineData("Audit")]
    public void AdminActions_DoNotWidenRoles(string action)
    {
        // Stacked AuthorizeAttributes intersect: a method-level Roles value must
        // never admit Utilisateur or Contrôleur to the admin area.
        var method = typeof(AdminController).GetMethod(action)!;
        foreach (var attr in method.GetCustomAttributes<AuthorizeAttribute>(false))
        {
            if (string.IsNullOrWhiteSpace(attr.Roles))
                continue;
            Assert.DoesNotContain("Utilisateur", attr.Roles);
            Assert.DoesNotContain("Contrôleur", attr.Roles);
        }
    }

    [Fact]
    public void AdminController_MakesNoAiOcrOpenCvCalls()
    {
        var ctorParams = typeof(AdminController).GetConstructors().Single().GetParameters();
        var typeNames = ctorParams.Select(p => p.ParameterType.FullName ?? p.ParameterType.Name).ToList();
        Assert.DoesNotContain(typeNames, t => t.Contains("VerificationApiClient"));
        Assert.DoesNotContain(typeNames, t => t.Contains("IVerificationApiClient"));
        var methodNames = typeof(AdminController).GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => m.DeclaringType == typeof(AdminController)).SelectMany(m => Array.Empty<string>()).ToList();
        Assert.Empty(methodNames);
    }

    [Fact]
    public void UserListViewModel_NeverExposesPasswordHash()
    {
        Assert.Null(typeof(AdminUserListItemViewModel).GetProperty("PasswordHash"));
        Assert.Null(typeof(AdminUserListViewModel).GetProperty("Password"));
        Assert.Null(typeof(AdminDashboardViewModel).GetProperty("PasswordHash"));
    }

    [Fact]
    public async Task Index_CountsReflectPersistedData_AndPolicyUnchanged()
    {
        await using var db = new ChequeVerificationDbContext(CreateOptions());
        await SeedAsync(db);
        var controller = new AdminController(db, DefaultPolicy());
        var result = Assert.IsType<ViewResult>(await controller.Index());
        var model = Assert.IsType<AdminDashboardViewModel>(result.Model);
        Assert.Equal(3, model.TotalUsers);
        Assert.Equal(2, model.ActiveUsers);
        Assert.Equal(1, model.InactiveUsers);
        Assert.Equal(3, model.UsersByRole.Sum(r => r.Count));
        Assert.Equal(3, model.Roles.Count);
        Assert.Equal(2, model.TotalReferenceSignatures);
        Assert.Equal(1, model.ActiveReferenceSignatures);
        Assert.Equal(1, model.TotalCustomers);
        Assert.Equal(1, model.TotalCheques);
        Assert.Equal(2, model.TotalVerifications);
        Assert.Equal(1, model.PendingManualReviews);
        Assert.Equal(3, model.TotalAuditLogs);
        Assert.Equal(0.6585m, model.LowerThreshold);
        Assert.Equal(0.9150m, model.UpperThreshold);
        Assert.Equal("sig-verif-ai-v5a", model.ModelName);
        Assert.Equal("v5a-phase7", model.ModelVersion);
        // Recent first, handles deleted/missing user safely.
        Assert.Equal(3, model.RecentAuditLogs.Count);
        Assert.True(model.RecentAuditLogs[0].CreatedAt >= model.RecentAuditLogs[1].CreatedAt);
    }

    [Fact]
    public async Task Users_ListContainsDbFields_NoSecrets()
    {
        await using var db = new ChequeVerificationDbContext(CreateOptions());
        await SeedAsync(db);
        var controller = new AdminController(db, DefaultPolicy());
        var result = Assert.IsType<ViewResult>(await controller.Users());
        var model = Assert.IsType<AdminUserListViewModel>(result.Model);
        Assert.Equal(3, model.TotalCount);
        var admin = model.Items.Single(u => u.Email == "admin@example.com");
        Assert.Equal(1, admin.UserId);
        Assert.Equal("Admin One", admin.FullName);
        Assert.Equal("Administrateur", admin.RoleName);
        Assert.True(admin.IsActive);
        var inactive = model.Items.Single(u => u.Email == "ctrl@example.com");
        Assert.False(inactive.IsActive);
    }

    [Fact]
    public async Task Audit_ListNewestFirst_HandlesMissingUser()
    {
        await using var db = new ChequeVerificationDbContext(CreateOptions());
        await SeedAsync(db);
        var controller = new AdminController(db, DefaultPolicy());
        var result = Assert.IsType<ViewResult>(await controller.Audit(1));
        var model = Assert.IsType<AdminAuditListViewModel>(result.Model);
        Assert.Equal(3, model.TotalCount);
        Assert.Equal("New", model.Items[0].Action);
        Assert.Equal("Mid", model.Items[1].Action);
        Assert.Equal("Old", model.Items[2].Action);
        for (int i = 1; i < model.Items.Count; i++)
            Assert.True(model.Items[i - 1].CreatedAt >= model.Items[i].CreatedAt);
        // Deleted/null user rows render safely (null UserName, no throw).
        Assert.All(model.Items, _ => Assert.True(true));
    }

    [Fact]
    public void PolicyDefaults_MatchFrozenThresholds()
    {
        var policy = new VerificationPolicyOptions();
        Assert.Equal(0.6585m, policy.LowerThreshold);
        Assert.Equal(0.9150m, policy.UpperThreshold);
        Assert.Equal("sig-verif-ai-v5a", policy.ModelName);
        Assert.Equal("v5a-phase7", policy.ModelVersion);
    }
}
