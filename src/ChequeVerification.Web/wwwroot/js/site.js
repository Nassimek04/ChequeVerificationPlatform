document.addEventListener('DOMContentLoaded', function () {
  // Presentation-only user initials derived from the already-rendered name.
  // No model/controller change; purely decorative.
  var nameEl = document.querySelector('.app-user-name');
  var avatarEl = document.getElementById('appUserAvatar');
  if (nameEl && avatarEl) {
    var parts = nameEl.textContent.trim().split(/\s+/).filter(Boolean);
    var initials = parts.slice(0, 2).map(function (p) { return p.charAt(0).toUpperCase(); }).join('');
    avatarEl.textContent = initials || '··';
  }

  const toggle = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('appSidebar');
  const backdrop = document.getElementById('appBackdrop');

  if (!toggle || !sidebar) {
    return;
  }

  const isDesktop = () => window.matchMedia('(min-width: 992px)').matches;

  function setOpen(open) {
    sidebar.classList.toggle('open', open);
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    toggle.setAttribute('aria-label', open ? 'Fermer le menu de navigation' : 'Ouvrir le menu de navigation');
    if (backdrop) {
      backdrop.classList.toggle('visible', open && !isDesktop());
    }
  }

  toggle.addEventListener('click', function () {
    setOpen(!sidebar.classList.contains('open'));
  });

  if (backdrop) {
    backdrop.addEventListener('click', function () {
      setOpen(false);
    });
  }

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && !isDesktop()) {
      setOpen(false);
    }
  });
});
