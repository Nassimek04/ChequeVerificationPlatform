using System.Security.Claims;
using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.ManualReviews;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ChequeVerification.Web.Controllers;

[Authorize(Roles = "Contrôleur")]
public class ManualReviewsController : Controller
{
    private readonly IManualReviewService _manualReviewService;

    public ManualReviewsController(IManualReviewService manualReviewService)
    {
        _manualReviewService = manualReviewService;
    }

    [HttpGet]
    public async Task<IActionResult> Queue(CancellationToken cancellationToken)
    {
        var items = await _manualReviewService.GetPendingQueueAsync(cancellationToken);
        var vm = new ManualReviewQueueViewModel { Items = items };
        return View(vm);
    }

    [HttpGet]
    public async Task<IActionResult> Review(int id, CancellationToken cancellationToken)
    {
        var vm = await _manualReviewService.GetReviewDetailsAsync(id, cancellationToken);
        if (vm == null)
        {
            return NotFound();
        }

        // Only allow reviewing pending manual cases
        if (vm.AutomaticDecision != 3 || vm.FinalDecision != null || vm.ChequeStatus != 4)
        {
            TempData["Error"] = "Cette vérification n'est plus en attente de contrôle manuel.";
            return RedirectToAction(nameof(Queue));
        }

        return View(vm);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Decide([Bind(Prefix = "Decision")] ManualReviewDecisionViewModel model, CancellationToken cancellationToken)
    {
        if (!ModelState.IsValid)
        {
            var vm = await _manualReviewService.GetReviewDetailsAsync(model.VerificationId, cancellationToken);
            if (vm == null) return NotFound();
            // Preserve user input
            vm.Decision = model;
            return View(nameof(Review), vm);
        }

        // Server-authoritative: FinalDecision must be 1 or 2, never trust 3
        if (model.FinalDecision != 1 && model.FinalDecision != 2)
        {
            ModelState.AddModelError(nameof(model.FinalDecision), "Décision invalide.");
            var vm = await _manualReviewService.GetReviewDetailsAsync(model.VerificationId, cancellationToken);
            if (vm == null) return NotFound();
            vm.Decision = model;
            return View(nameof(Review), vm);
        }

        var userIdClaim = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdClaim, out var userId))
        {
            return Forbid();
        }

        var result = await _manualReviewService.DecideAsync(
            model.VerificationId,
            model.FinalDecision.Value,
            model.ReviewerComment,
            userId,
            cancellationToken);

        if (!result.Success)
        {
            ModelState.AddModelError(string.Empty, result.Message);
            var vm = await _manualReviewService.GetReviewDetailsAsync(model.VerificationId, cancellationToken);
            if (vm == null) return NotFound();
            // If already reviewed, redirect to queue with error
            if (result.Message.Contains("déjà été traitée") || result.Message.Contains("n'est plus en attente"))
            {
                TempData["Error"] = result.Message;
                return RedirectToAction(nameof(Queue));
            }
            vm.Decision = model;
            return View(nameof(Review), vm);
        }

        TempData["Success"] = result.Message;
        return RedirectToAction(nameof(Queue));
    }
}
