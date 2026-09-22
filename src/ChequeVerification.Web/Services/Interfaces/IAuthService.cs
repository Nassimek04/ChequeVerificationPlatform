using System.Security.Claims;
using ChequeVerification.Web.Models.Entities;

namespace ChequeVerification.Web.Services.Interfaces;

public interface IAuthService
{
    Task<User?> AuthenticateAsync(string email, string password);

    ClaimsPrincipal CreatePrincipal(User user);

    Task UpdateLastLoginAsync(User user);

    Task LogAuditAsync(int? userId, string action, string entityName, int? entityId, string? description);
}
