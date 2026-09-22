using System.Diagnostics;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using ChequeVerification.Web.Models;
using ChequeVerification.Web.Services.Interfaces;

namespace ChequeVerification.Web.Controllers;

[Authorize]
public class HomeController : Controller
{
    private readonly IDashboardService _dashboardService;

    public HomeController(IDashboardService dashboardService)
    {
        _dashboardService = dashboardService;
    }

    public async Task<IActionResult> Index()
    {
        if (User.IsInRole("Administrateur"))
        {
            return View("Admin", await _dashboardService.GetAdminDashboardAsync());
        }

        if (User.IsInRole("Contrôleur"))
        {
            return View("Controller", await _dashboardService.GetControllerDashboardAsync());
        }

        return View("User", await _dashboardService.GetUserDashboardAsync());
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}