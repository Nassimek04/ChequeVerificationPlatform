using System.Security.Claims;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ChequeVerification.Web.Controllers;

[AllowAnonymous]
public class AccountController : Controller
{
    private readonly IAuthService _authService;
    private readonly ILogger<AccountController> _logger;

    public AccountController(IAuthService authService, ILogger<AccountController> logger)
    {
        _authService = authService;
        _logger = logger;
    }

    [HttpGet]
    public IActionResult Login(string? returnUrl = null)
    {
        if (User.Identity?.IsAuthenticated == true)
        {
            return RedirectToLocal(returnUrl);
        }

        var model = new LoginViewModel { ReturnUrl = returnUrl };
        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Login(LoginViewModel model)
    {
        if (!ModelState.IsValid)
        {
            return View(model);
        }

        var user = await _authService.AuthenticateAsync(model.Email, model.Password);

        if (user == null)
        {
            _logger.LogWarning("Échec d'authentification pour {Email}", model.Email);
            ModelState.AddModelError(string.Empty, "Email ou mot de passe incorrect.");
            return View(model);
        }

        var principal = _authService.CreatePrincipal(user);

        await HttpContext.SignInAsync(
            CookieAuthenticationDefaults.AuthenticationScheme,
            principal,
            new AuthenticationProperties
            {
                IsPersistent = model.RememberMe,
                ExpiresUtc = model.RememberMe ? DateTimeOffset.UtcNow.AddDays(7) : null
            });

        await _authService.UpdateLastLoginAsync(user);
        await _authService.LogAuditAsync(user.UserId, "Login", "User", user.UserId, $"Connexion réussie : {user.Email}");

        _logger.LogInformation("Connexion réussie : {Email}", user.Email);

        return RedirectToLocal(model.ReturnUrl);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Logout()
    {
        var userIdClaim = User.FindFirstValue(ClaimTypes.NameIdentifier);
        int? userId = int.TryParse(userIdClaim, out var id) ? id : null;

        await HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);

        if (userId.HasValue)
        {
            await _authService.LogAuditAsync(userId, "Logout", "User", userId, "Déconnexion");
        }

        return RedirectToAction(nameof(Login));
    }

    [HttpGet]
    public IActionResult AccessDenied()
    {
        return View();
    }

    private IActionResult RedirectToLocal(string? returnUrl)
    {
        if (!string.IsNullOrWhiteSpace(returnUrl) && Url.IsLocalUrl(returnUrl))
        {
            return LocalRedirect(returnUrl);
        }

        return RedirectToAction(nameof(HomeController.Index), "Home");
    }
}
