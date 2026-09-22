using System.Reflection;
using ChequeVerification.Web.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ChequeVerification.Web.Tests;

// Guards the authorization structure of VerificationsController.
//
// ASP.NET Core COMBINES every AuthorizeAttribute on the controller and the
// action (all policies must pass). A restrictive class-level Roles value
// therefore overrides a wider action-level policy instead of being
// replaced by it. These tests evaluate the EFFECTIVE (stacked) policy.
public class VerificationAuthorizationTests
{
    private static readonly HashSet<string> UtilAdmin = new() { "Utilisateur", "Administrateur" };
    private static readonly HashSet<string> AllThree = new() { "Utilisateur", "Administrateur", "Contrôleur" };

    [Fact]
    public void ControllerClass_DoesNotRestrictRoles()
    {
        // Plain [Authorize]: authentication only. Role restriction lives on
        // each action explicitly so History/Details can admit Contrôleur.
        var classAuth = typeof(VerificationsController)
            .GetCustomAttributes<AuthorizeAttribute>(false).ToList();
        Assert.NotEmpty(classAuth);
        Assert.All(classAuth, a => Assert.True(string.IsNullOrWhiteSpace(a.Roles)));
        Assert.Empty(typeof(VerificationsController)
            .GetCustomAttributes<AllowAnonymousAttribute>(false));
    }

    [Theory]
    [InlineData("History")]
    [InlineData("Details")]
    [InlineData("DetailsByCheque")]
    public void ReadOnlyActions_AllowAllThreeRoles(string action)
    {
        Assert.Equal(AllThree, EffectiveRoles(typeof(VerificationsController), action));
    }

    [Theory]
    [InlineData("Create")]
    [InlineData("TestImage")]
    [InlineData("TestSignatureExtraction")]
    [InlineData("DebugSignatureExtraction")]
    [InlineData("ExtractSignature")]
    [InlineData("ReExtractSignature")]
    [InlineData("TestSignatureComparison")]
    [InlineData("TestSignatureComparisonAi")]
    [InlineData("TestChequeOcr")]
    [InlineData("LaunchVerification")]
    public void MutationAndTestActions_DenyControleur(string action)
    {
        var roles = EffectiveRoles(typeof(VerificationsController), action);
        Assert.Equal(UtilAdmin, roles);
        Assert.DoesNotContain("Contrôleur", roles);
    }

    [Fact]
    public void Create_HasExplicitMethodLevelRoles()
    {
        // Create relied on class-level inheritance before the fix; it must
        // now carry its own Roles so a future class-level change cannot
        // silently broaden it.
        var method = typeof(VerificationsController).GetMethod("Create")!;
        var roles = MethodRoles(method);
        Assert.Equal(UtilAdmin, roles);
    }

    [Fact]
    public void EveryAction_RequiresAuthentication()
    {
        var actions = typeof(VerificationsController)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => m.ReturnType == typeof(IActionResult)
                || (m.ReturnType.IsGenericType
                    && m.ReturnType.GetGenericTypeDefinition() == typeof(Task<>)
                    && typeof(IActionResult).IsAssignableFrom(m.ReturnType.GetGenericArguments()[0]))
                && m.DeclaringType == typeof(VerificationsController));
        Assert.NotEmpty(actions);
        foreach (var action in actions)
        {
            var hasAuth = action.GetCustomAttributes<AuthorizeAttribute>(false).Any()
                || typeof(VerificationsController).GetCustomAttributes<AuthorizeAttribute>(false).Any();
            Assert.True(hasAuth, $"{action.Name} must require authentication.");
            Assert.Empty(action.GetCustomAttributes<AllowAnonymousAttribute>(false));
        }
    }

    [Theory]
    [InlineData("Queue")]
    [InlineData("Review")]
    [InlineData("Decide")]
    public void ManualReviews_RemainControleurOnly(string action)
    {
        Assert.Equal(new HashSet<string> { "Contrôleur" },
            EffectiveRoles(typeof(ManualReviewsController), action));
    }

    // Mimics ASP.NET Core default stacking: every AuthorizeAttribute
    // (controller + action) must be satisfied, so the effective role set is
    // the INTERSECTION of all Roles requirements. An attribute without Roles
    // (plain authentication) is neutral for the intersection.
    private static HashSet<string> EffectiveRoles(Type controller, string actionName)
    {
        var method = controller.GetMethod(actionName)
            ?? throw new InvalidOperationException($"Action {controller.Name}.{actionName} not found.");
        HashSet<string>? effective = null;
        foreach (var attr in controller.GetCustomAttributes<AuthorizeAttribute>(false)
                     .Concat(method.GetCustomAttributes<AuthorizeAttribute>(false)))
        {
            if (string.IsNullOrWhiteSpace(attr.Roles))
                continue;
            var roles = attr.Roles.Split(',').Select(r => r.Trim())
                .Where(r => r.Length > 0).ToHashSet();
            effective = effective == null ? roles : new HashSet<string>(effective.Intersect(roles));
        }
        return effective ?? new HashSet<string>();
    }

    private static HashSet<string> MethodRoles(MethodInfo method)
    {
        var attr = method.GetCustomAttributes<AuthorizeAttribute>(false).FirstOrDefault();
        Assert.NotNull(attr);
        return attr!.Roles!.Split(',').Select(r => r.Trim()).ToHashSet();
    }
}
