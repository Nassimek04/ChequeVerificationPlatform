using System.Security.Claims;
using ChequeVerification.Web.Data;
using ChequeVerification.Web.Models.Entities;
using ChequeVerification.Web.Services.Interfaces;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Services;

public class AuthService : IAuthService
{
    private const byte UserStatusActive = 1;

    private readonly ChequeVerificationDbContext _db;
    private readonly IPasswordHasher<User> _passwordHasher;
    private readonly ILogger<AuthService> _logger;

    public AuthService(ChequeVerificationDbContext db, IPasswordHasher<User> passwordHasher, ILogger<AuthService> logger)
    {
        _db = db;
        _passwordHasher = passwordHasher;
        _logger = logger;
    }

    public async Task<User?> AuthenticateAsync(string email, string password)
    {
        var user = await _db.Users
            .Include(u => u.Role)
            .AsNoTracking()
            .SingleOrDefaultAsync(u => u.Email == email);

        if (user == null || user.Role == null)
        {
            return null;
        }

        if (user.Status != UserStatusActive)
        {
            _logger.LogWarning("Tentative de connexion pour un compte inactif : {Email}", email);
            return null;
        }

        var verificationResult = _passwordHasher.VerifyHashedPassword(user, user.PasswordHash, password);

        if (verificationResult == PasswordVerificationResult.SuccessRehashNeeded)
        {
            _logger.LogInformation("Le hash de {Email} doit être régénéré (sera géré ultérieurement)", email);
            return user;
        }

        return verificationResult == PasswordVerificationResult.Success ? user : null;
    }

    public ClaimsPrincipal CreatePrincipal(User user)
    {
        var claims = new List<Claim>
        {
            new(ClaimTypes.NameIdentifier, user.UserId.ToString()),
            new(ClaimTypes.Name, user.FullName),
            new(ClaimTypes.Email, user.Email),
            new(ClaimTypes.Role, user.Role!.Name)
        };

        var identity = new ClaimsIdentity(claims, "ApplicationCookie");
        return new ClaimsPrincipal(identity);
    }

    public async Task UpdateLastLoginAsync(User user)
    {
        var tracked = await _db.Users.FindAsync(user.UserId);
        if (tracked != null)
        {
            tracked.LastLogin = DateTime.UtcNow;
            await _db.SaveChangesAsync();
        }
    }

    public async Task LogAuditAsync(int? userId, string action, string entityName, int? entityId, string? description)
    {
        try
        {
            _db.AuditLogs.Add(new AuditLog
            {
                UserId = userId,
                Action = action,
                EntityName = entityName,
                EntityId = entityId,
                Description = description,
                CreatedAt = DateTime.UtcNow
            });
            await _db.SaveChangesAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Impossible d'enregistrer l'entrée d'audit pour {Action}", action);
        }
    }
}
