(() => {
  const storageKey = 'aiinffra-theme';
  const themes = new Set(['light', 'dark']);

  function readStoredTheme() {
    try {
      const value = window.localStorage.getItem(storageKey);
      return themes.has(value) ? value : null;
    } catch {
      return null;
    }
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    const button = document.querySelector('#theme-toggle');
    if (!button) return;
    const next = theme === 'dark' ? 'light' : 'dark';
    button.textContent = next === 'dark' ? '暗色' : '浅色';
    button.setAttribute('aria-label', `切换到${next === 'dark' ? '暗色' : '浅色'}模式`);
    button.title = `切换到${next === 'dark' ? '暗色' : '浅色'}模式`;
    button.setAttribute('aria-pressed', String(theme === 'dark'));
  }

  function saveTheme(theme) {
    try { window.localStorage.setItem(storageKey, theme); } catch {}
  }

  // Runs synchronously in <head> so the first stylesheet sees the final theme.
  applyTheme(readStoredTheme() || 'dark');

  document.addEventListener('DOMContentLoaded', () => {
    const button = document.querySelector('#theme-toggle');
    if (!button || button.dataset.themeBound) return;
    button.dataset.themeBound = 'true';
    button.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      saveTheme(next);
    });
    applyTheme(document.documentElement.dataset.theme || 'dark');
  }, {once: true});
})();
