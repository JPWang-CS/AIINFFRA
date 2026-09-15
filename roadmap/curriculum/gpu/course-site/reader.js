document.querySelectorAll('.copy-code').forEach(button => {
  const source = button.closest('.code-panel').querySelector('code').textContent;
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(source);
      button.textContent = '已复制';
    } catch {
      const area = document.createElement('textarea');
      area.value = source; area.style.position = 'fixed'; area.style.opacity = '0';
      document.body.append(area); area.select();
      try { document.execCommand('copy'); button.textContent = '已复制'; }
      catch { button.textContent = '请选中代码复制'; }
      area.remove();
    }
    setTimeout(() => { button.textContent = '复制'; }, 1600);
  });
});
