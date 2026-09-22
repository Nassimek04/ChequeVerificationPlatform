using ChequeVerification.Web.ViewModels.Dashboard;

namespace ChequeVerification.Web.Services.Interfaces;

public interface IDashboardService
{
    Task<UserDashboardViewModel> GetUserDashboardAsync();
    Task<ControllerDashboardViewModel> GetControllerDashboardAsync();
    Task<AdminDashboardViewModel> GetAdminDashboardAsync();
}