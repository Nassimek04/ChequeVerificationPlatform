using ChequeVerification.Web.Data;
using ChequeVerification.Web.Services;
using ChequeVerification.Web.ViewModels.Admin;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace ChequeVerification.Web.Controllers;

[Authorize(Roles = "Administrateur")]
public class AdminController : Controller
{
    private const byte UserStatusActive = 1;

    private readonly ChequeVerificationDbContext _db;
    private readonly VerificationPolicyOptions _policy;

    // Read-only admin area: DB + policy snapshot only. No AI/OCR/OpenCV client.
    public AdminController(ChequeVerificationDbContext db, IOptions<VerificationPolicyOptions> policy)
    {
        _db = db;
        _policy = policy.Value;
    }

    [HttpGet]
    public async Task<IActionResult> Index(CancellationToken cancellationToken = default)
    {
        var usersByRole = await _db.Users
            .AsNoTracking()
            .Include(u => u.Role)
            .GroupBy(u => u.Role!.Name)
            .Select(g => new AdminUsersByRoleViewModel { RoleName = g.Key, Count = g.Count() })
            .OrderBy(r => r.RoleName)
            .ToListAsync(cancellationToken);

        var roles = await _db.Roles
            .AsNoTracking()
            .OrderBy(r => r.Name)
            .Select(r => new AdminRoleViewModel
            {
                RoleId = r.RoleId,
                Name = r.Name,
                Description = r.Description,
                UserCount = r.Users.Count
            })
            .ToListAsync(cancellationToken);

        var recentAuditLogs = await _db.AuditLogs
            .AsNoTracking()
            .OrderByDescending(a => a.CreatedAt)
            .ThenByDescending(a => a.AuditLogId)
            .Take(8)
            .Select(a => new AdminAuditItemViewModel
            {
                AuditLogId = a.AuditLogId,
                CreatedAt = a.CreatedAt,
                UserName = a.User != null ? a.User.FullName : null,
                Action = a.Action,
                EntityName = a.EntityName,
                EntityId = a.EntityId,
                Description = a.Description
            })
            .ToListAsync(cancellationToken);

        var model = new AdminDashboardViewModel
        {
            TotalUsers = await _db.Users.AsNoTracking().CountAsync(cancellationToken),
            ActiveUsers = await _db.Users.AsNoTracking().CountAsync(u => u.Status == UserStatusActive, cancellationToken),
            InactiveUsers = await _db.Users.AsNoTracking().CountAsync(u => u.Status != UserStatusActive, cancellationToken),
            UsersByRole = usersByRole,
            Roles = roles,
            TotalReferenceSignatures = await _db.ReferenceSignatures.AsNoTracking().CountAsync(cancellationToken),
            ActiveReferenceSignatures = await _db.ReferenceSignatures.AsNoTracking().CountAsync(r => r.IsActive, cancellationToken),
            LowerThreshold = _policy.LowerThreshold,
            UpperThreshold = _policy.UpperThreshold,
            ModelName = _policy.ModelName,
            ModelVersion = _policy.ModelVersion,
            TotalCustomers = await _db.Customers.AsNoTracking().CountAsync(cancellationToken),
            TotalCheques = await _db.Cheques.AsNoTracking().CountAsync(cancellationToken),
            TotalVerifications = await _db.VerificationResults.AsNoTracking().CountAsync(cancellationToken),
            PendingManualReviews = await _db.VerificationResults.AsNoTracking().CountAsync(v => v.FinalDecision == null, cancellationToken),
            TotalAuditLogs = await _db.AuditLogs.AsNoTracking().LongCountAsync(cancellationToken),
            RecentAuditLogs = recentAuditLogs
        };

        return View(model);
    }

    [HttpGet]
    public async Task<IActionResult> Users(CancellationToken cancellationToken = default)
    {
        var items = await _db.Users
            .AsNoTracking()
            .Include(u => u.Role)
            .OrderBy(u => u.FullName)
            .Select(u => new AdminUserListItemViewModel
            {
                UserId = u.UserId,
                FullName = u.FullName,
                Email = u.Email,
                RoleName = u.Role!.Name,
                Status = u.Status,
                IsActive = u.Status == UserStatusActive,
                CreatedAt = u.CreatedAt,
                LastLogin = u.LastLogin
            })
            .ToListAsync(cancellationToken);

        return View(new AdminUserListViewModel { Items = items, TotalCount = items.Count });
    }

    [HttpGet]
    public async Task<IActionResult> Audit(int page = 1, CancellationToken cancellationToken = default)
    {
        const int pageSize = 20;
        if (page < 1)
        {
            page = 1;
        }

        var totalCount = await _db.AuditLogs.AsNoTracking().LongCountAsync(cancellationToken);

        var items = await _db.AuditLogs
            .AsNoTracking()
            .OrderByDescending(a => a.CreatedAt)
            .ThenByDescending(a => a.AuditLogId)
            .Skip((page - 1) * pageSize)
            .Take(pageSize)
            .Select(a => new AdminAuditItemViewModel
            {
                AuditLogId = a.AuditLogId,
                CreatedAt = a.CreatedAt,
                UserName = a.User != null ? a.User.FullName : null,
                Action = a.Action,
                EntityName = a.EntityName,
                EntityId = a.EntityId,
                Description = a.Description
            })
            .ToListAsync(cancellationToken);

        return View(new AdminAuditListViewModel
        {
            Items = items,
            Page = page,
            PageSize = pageSize,
            TotalCount = totalCount
        });
    }
}
