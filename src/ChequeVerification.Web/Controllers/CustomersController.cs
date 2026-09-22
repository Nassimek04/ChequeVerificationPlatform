using ChequeVerification.Web.Services.Interfaces;
using ChequeVerification.Web.ViewModels.Customers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ChequeVerification.Web.Controllers;

[Authorize]
public class CustomersController : Controller
{
    private readonly ICustomerService _customerService;

    public CustomersController(ICustomerService customerService)
    {
        _customerService = customerService;
    }

    [HttpGet]
    public async Task<IActionResult> Index(string? search, int page = 1)
    {
        var model = await _customerService.GetPagedAsync(search, page, 10);
        return View(model);
    }

    [HttpGet]
    public async Task<IActionResult> Details(int id)
    {
        var model = await _customerService.GetDetailAsync(id);

        if (model == null)
        {
            return NotFound();
        }

        return View(model);
    }

}
