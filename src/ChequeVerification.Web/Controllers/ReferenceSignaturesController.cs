using System.Security.Claims;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.ReferenceSignatures;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ChequeVerification.Web.Controllers;

[Authorize]
public class ReferenceSignaturesController : Controller
{
    private readonly IReferenceSignatureService _referenceSignatureService;
    private readonly IWebHostEnvironment _environment;

    public ReferenceSignaturesController(IReferenceSignatureService referenceSignatureService, IWebHostEnvironment environment)
    {
        _referenceSignatureService = referenceSignatureService;
        _environment = environment;
    }

    [HttpGet]
    public async Task<IActionResult> Index(string? search, int page = 1, CancellationToken cancellationToken = default)
    {
        var model = await _referenceSignatureService.GetPagedAsync(search, page, 10, cancellationToken);
        return View(model);
    }

    [HttpGet]
    public async Task<IActionResult> Details(int id, CancellationToken cancellationToken = default)
    {
        var model = await _referenceSignatureService.GetDetailsAsync(id, cancellationToken);

        if (model == null)
        {
            return NotFound();
        }

        model.ImageIsAccessible = ResolveImageIsAccessible(model.ImagePath);
        return View(model);
    }

    [HttpGet]
    [Authorize(Roles = "Administrateur")]
    public async Task<IActionResult> Create(int customerId, CancellationToken cancellationToken = default)
    {
        var model = await _referenceSignatureService.GetCreateViewModelAsync(customerId, cancellationToken);

        if (model == null)
        {
            return NotFound();
        }

        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Administrateur")]
    public async Task<IActionResult> Create(ReferenceSignatureCreateViewModel model, CancellationToken cancellationToken = default)
    {
        // Reload customer info for redisplay on validation failure.
        var vm = await _referenceSignatureService.GetCreateViewModelAsync(model.CustomerId, cancellationToken);
        if (vm == null)
        {
            return NotFound();
        }

        vm.ImageFile = model.ImageFile;

        if (model.ImageFile == null)
        {
            ModelState.AddModelError(nameof(model.ImageFile), "L'image de la signature est obligatoire.");
        }

        if (!ModelState.IsValid)
        {
            return View(vm);
        }

        var result = await _referenceSignatureService.CreateAsync(
            model.CustomerId,
            model.ImageFile!,
            GetCurrentUserId(),
            _environment.WebRootPath,
            cancellationToken);

        if (!result.Success)
        {
            ModelState.AddModelError(string.Empty, result.Message);
            // Refresh active count (in case cap/duplicate message depends on latest count).
            var refreshed = await _referenceSignatureService.GetCreateViewModelAsync(model.CustomerId, cancellationToken);
            if (refreshed != null)
            {
                refreshed.ImageFile = null;
                return View(refreshed);
            }

            return View(vm);
        }

        TempData["Success"] = $"Signature de référence #{result.ReferenceSignatureId} enrôlée avec succès.";
        return RedirectToAction("Details", "Customers", new { id = model.CustomerId });
    }

    private int? GetCurrentUserId()
    {
        var value = User.FindFirstValue(ClaimTypes.NameIdentifier);
        return int.TryParse(value, out var id) ? id : null;
    }

    private bool ResolveImageIsAccessible(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            return false;
        }

        if (imagePath.StartsWith("http://") || imagePath.StartsWith("https://"))
        {
            return false;
        }

        try
        {
            var localPath = imagePath.StartsWith("/") || imagePath.StartsWith("\\")
                ? Path.Combine(_environment.WebRootPath, imagePath.TrimStart('/', '\\').Replace('/', Path.DirectorySeparatorChar))
                : imagePath;

            return System.IO.File.Exists(localPath);
        }
        catch
        {
            return false;
        }
    }
}