using System.Security.Claims;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Cheques;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ChequeVerification.Web.Controllers;

[Authorize]
public class ChequesController : Controller
{
    private readonly IChequeService _chequeService;
    private readonly IChequeBatchImportService _batchImportService;
    private readonly IWebHostEnvironment _environment;
    private readonly ILogger<ChequesController> _logger;

    public ChequesController(IChequeService chequeService, IChequeBatchImportService batchImportService, IWebHostEnvironment environment, ILogger<ChequesController> logger)
    {
        _chequeService = chequeService;
        _batchImportService = batchImportService;
        _environment = environment;
        _logger = logger;
    }

    [HttpGet]
    public async Task<IActionResult> Index(string? search, int page = 1, CancellationToken cancellationToken = default)
    {
        var model = await _chequeService.GetPagedAsync(search, page, 10, cancellationToken);
        return View(model);
    }

    [HttpGet]
    public async Task<IActionResult> Details(int id, CancellationToken cancellationToken = default)
    {
        var model = await _chequeService.GetDetailsAsync(id, cancellationToken);

        if (model == null)
        {
            return NotFound();
        }

        model.ImageIsAccessible = ResolveImageIsAccessible(model.ImagePath);
        return View(model);
    }

    [HttpGet]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> Create(CancellationToken cancellationToken = default)
    {
        var model = new ChequeCreateViewModel
        {
            ChequeNumber = await _chequeService.GenerateChequeNumberAsync(cancellationToken),
            Customers = await _chequeService.GetCustomerOptionsAsync(cancellationToken)
        };

        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> Create(ChequeCreateViewModel model, CancellationToken cancellationToken = default)
    {
        model.Customers = await _chequeService.GetCustomerOptionsAsync(cancellationToken);

        if (model.ImageFile == null)
        {
            ModelState.AddModelError(nameof(model.ImageFile), "L'image du chèque est obligatoire.");
        }

        if (!ModelState.IsValid)
        {
            return View(model);
        }

        var userId = GetCurrentUserId();
        if (userId <= 0)
        {
            return Forbid();
        }

        try
        {
            var cheque = await _chequeService.CreateAsync(model, userId, _environment.WebRootPath, cancellationToken);
            TempData["Success"] = $"Chèque {cheque.ChequeNumber} importé avec succès.";
            return RedirectToAction(nameof(Details), new { id = cheque.ChequeId });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Échec de l'import du chèque.");
            ModelState.AddModelError(string.Empty, ex.Message);
            return View(model);
        }
    }

    private int GetCurrentUserId()
    {
        var value = User.FindFirstValue(ClaimTypes.NameIdentifier);
        return int.TryParse(value, out var id) ? id : 0;
    }

    // ---- Phase 1 : import par lot (même autorisation que l'import unitaire) ----

    [HttpGet]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public IActionResult BatchImport()
    {
        return View(new BatchImportUploadViewModel());
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    [RequestSizeLimit(110_000_000)]
    public async Task<IActionResult> BatchImport(List<IFormFile> files, CancellationToken cancellationToken = default)
    {
        var preview = await _batchImportService.BuildPreviewAsync(
            files ?? new List<IFormFile>(), _environment.WebRootPath, cancellationToken);

        if (!string.IsNullOrEmpty(preview.GlobalError))
        {
            ModelState.AddModelError(string.Empty, preview.GlobalError);
            return View(new BatchImportUploadViewModel());
        }

        return View("BatchPreview", preview);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [Authorize(Roles = "Utilisateur,Administrateur")]
    public async Task<IActionResult> ConfirmBatchImport(BatchImportConfirmViewModel model, CancellationToken cancellationToken = default)
    {
        var userId = GetCurrentUserId();
        if (userId <= 0)
        {
            return Forbid();
        }

        var result = await _batchImportService.ConfirmImportAsync(
            model ?? new BatchImportConfirmViewModel(), userId, _environment.WebRootPath, cancellationToken);
        return View("BatchResult", result);
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