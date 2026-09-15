const sidebar = document.querySelector('.sidebar');
const articles = [...document.querySelectorAll('.chapter')];
const articleById = new Map(articles.map(article => [Number(article.dataset.chapter), article]));
const paperDataElement = document.querySelector('#paper-content-data');
const paperContents = paperDataElement ? JSON.parse(paperDataElement.textContent) : {};
let navigationRevision = 0;
let interactionRevision = 0;
for (const type of ['wheel', 'touchstart', 'pointerdown', 'keydown']) {
  addEventListener(type, () => { interactionRevision++; }, {passive:true});
}
function settleAnchorAfterFonts(anchor) {
  if (!anchor || !document.fonts || document.fonts.status !== 'loading') return;
  const navigation = navigationRevision;
  const interaction = interactionRevision;
  document.fonts.ready.then(() => requestAnimationFrame(() => {
    // Font metrics can move a late-hydrated equation. Do not override a newer
    // navigation or any user interaction while the fonts were loading.
    if (navigation === navigationRevision && interaction === interactionRevision
        && anchor.isConnected && !anchor.closest('.chapter')?.hidden) {
      anchor.scrollIntoView({behavior:'instant'});
    }
  })).catch(() => {});
}
function bindCopies(root) {
  root.querySelectorAll('.code-panel').forEach(panel => {
    const scrollBox = panel.querySelector('pre');
    if (scrollBox && panel.closest('details')) {
      scrollBox.tabIndex = 0;
      scrollBox.setAttribute('aria-label', '代码区域，可用方向键滚动');
    }
    const button = panel.querySelector('.copy-code');
    if (!button || button.dataset.copyBound) return;
    button.dataset.copyBound = 'true';
    button.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(panel.querySelector('code').textContent); button.textContent = '已复制'; }
      catch { button.textContent = '请选中代码复制'; }
      setTimeout(() => button.textContent = '复制', 1600);
    });
  });
}
function hydratePaper(article) {
  const content = article?.querySelector('[data-paper-pending]');
  if (!content) return;
  content.innerHTML = paperContents[article.dataset.chapter] || '';
  content.removeAttribute('data-paper-pending');
  bindCopies(content);
}
const chapterLinks = [...document.querySelectorAll('[data-chapter-link]')];
const partLinks = [...document.querySelectorAll('[data-part-link]')];
const sidebarParts = [...document.querySelectorAll('[data-sidebar-part]')];
const navChapters = [...document.querySelectorAll('[data-nav-chapter]')];
const toc = document.querySelector('#page-toc');
const menu = document.querySelector('#menu-button');
const search = document.querySelector('#search');
const results = document.querySelector('#search-results');
const chapterNav = document.querySelector('#chapter-nav');
if (chapterNav && results) chapterNav.parentElement.insertBefore(results, chapterNav);
const searchPayload = document.querySelector('#course-search-data');
const searchIndex = searchPayload ? JSON.parse(searchPayload.textContent) : articles.map((article, i) => ({chapter:Number(article.dataset.chapter), title:article.dataset.title, text:article.textContent}));
let current = 1;
let headings = [];
const firstChapterByPart = new Map(partLinks.map(link => [link.dataset.partLink, Number(link.dataset.chapterLink)]));
const lastChapterByPart = new Map(firstChapterByPart);
const partOf = chapter => articleById.get(chapter)?.dataset.part || (chapter <= 5 ? 'gpu' : chapter <= 15 ? 'operators' : 'systems');
function readLocation() {
  const url = new URL(location.href);
  const chapter = Number(url.searchParams.get('chapter'));
  if (chapter === 6) return {chapter:7, hash:''};
  return {chapter: articleById.has(chapter) ? chapter : 1, hash: url.hash};
}
function clearSearch() {
  const wasSearching = !!search?.value.trim() || !!(results && !results.hidden);
  if (search) search.value = '';
  if (results) {
    results.replaceChildren();
    results.hidden = true;
  }
  if (chapterNav) chapterNav.hidden = false;
  if (sidebar && wasSearching) sidebar.scrollTop = 0;
}
function navigate(target, {replace = false, scroll = true} = {}) {
  const url = target instanceof URL ? target : new URL(target, location.href);
  if (!url.searchParams.has('chapter')) url.searchParams.set('chapter', current);
  if (replace) history.replaceState(null, '', url);
  else history.pushState(null, '', url);
  applyState(readLocation(), {scroll});
}
function closeMenu() { sidebar.classList.remove('open'); menu.setAttribute('aria-expanded', 'false'); }
let lastPanelSelection = '';
function revealInPanel(item, panel) {
  if (!item || !panel || !item.getClientRects().length || !panel.getClientRects().length) return;
  const row = item.getBoundingClientRect(), frame = panel.getBoundingClientRect();
  const inset = 12;
  // Scroll only the fixed panel; scrollIntoView could also move the article.
  if (row.top < frame.top + inset) panel.scrollTop += row.top - frame.top - inset;
  else if (row.bottom > frame.bottom - inset) panel.scrollTop += row.bottom - frame.bottom + inset;
}
function progress() {
  const d = document.documentElement;
  document.querySelector('#progress-bar').style.width = 100 * d.scrollTop / Math.max(1, d.scrollHeight - d.clientHeight) + '%';
  let selected = headings[0];
  for (const heading of headings) {
    if (heading.getBoundingClientRect().top <= 130) selected = heading;
    else break;
  }
  toc.querySelectorAll('a').forEach(link => {
    const active = !!selected && link.hash === '#' + selected.id;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  document.querySelectorAll('[data-section-link]').forEach(link => {
    const active = !!selected && link.dataset.sectionLink === selected.id;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  const panelSelection = selected ? navigationRevision + ':' + selected.id : '';
  if (panelSelection && panelSelection !== lastPanelSelection) {
    lastPanelSelection = panelSelection;
    revealInPanel(toc.querySelector('a.active'), toc.closest('.toc'));
    revealInPanel(document.querySelector('[data-section-link].active'), sidebar);
  }
}
function applyState(state, {scroll = true} = {}) {
  navigationRevision++;
  clearSearch();
  const n = state.chapter;
  hydratePaper(articleById.get(n));
  const normalizedUrl = new URL(location.href);
  if (normalizedUrl.searchParams.get('chapter') !== String(n)) {
    normalizedUrl.searchParams.set('chapter', n);
    normalizedUrl.hash = state.hash;
    history.replaceState(null, '', normalizedUrl);
  }
  current = n;
  lastChapterByPart.set(partOf(n), n);
  articles.forEach(article => article.hidden = Number(article.dataset.chapter) !== n);
  chapterLinks.forEach(link => {
    const active = Number(link.dataset.chapterLink) === n;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  document.querySelectorAll('[data-part-link]').forEach(link => {
    const active = link.dataset.partLink === partOf(n);
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  sidebarParts.forEach(panel => panel.hidden = panel.dataset.sidebarPart !== partOf(n));
  navChapters.forEach(details => {
    const active = Number(details.dataset.navChapter) === n;
    details.classList.toggle('active', active);
    if (active) {
      details.open = true;
      const category = details.closest('.paper-category');
      if (category) category.open = true;
    }
  });
  document.title = articleById.get(n).dataset.title + ' · AIINFFRA';
  headings = [...articleById.get(n).querySelectorAll('.markdown-body h2,.markdown-body h3')];
  headings.forEach((heading, i) => { if (!heading.id) heading.id = 'chapter-' + n + '-section-' + i; });
  toc.replaceChildren();
  for (const heading of headings) {
    const link = document.createElement('a');
    link.href = '?chapter=' + n + '#' + heading.id;
    link.textContent = heading.textContent.replace(/#$/, '');
    if (heading.tagName === 'H3') link.className = 'level-3';
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      navigate(new URL(link.href, location.href), {scroll: false});
      heading.scrollIntoView({behavior: 'smooth'});
      settleAnchorAfterFonts(heading);
    });
    toc.append(link);
  }
  let anchor;
  try { anchor = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch {}
  if (scroll) {
    if (anchor && articleById.get(n).contains(anchor)) {
      anchor.scrollIntoView({behavior:'instant'});
      settleAnchorAfterFonts(anchor);
    }
    else window.scrollTo({top:0, left:0, behavior:'instant'});
  }
  closeMenu(); progress();
}
chapterLinks.filter(link => !link.hasAttribute('data-section-link') && !link.hasAttribute('data-part-link')).forEach(link => link.addEventListener('click', event => {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  const url = new URL(link.href, location.href); url.hash = '';
  navigate(url);
}));
document.querySelectorAll('.chapter-outline a,.heading-anchor,[data-section-link]').forEach(link => link.addEventListener('click', event => {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  const id = link.hash.slice(1);
  const heading = document.getElementById(id);
  if (!heading) return;
  event.preventDefault();
  navigate(new URL(link.href, location.href), {scroll: false});
  const hydratedHeading = document.getElementById(id);
  hydratedHeading?.scrollIntoView({behavior: 'smooth'});
  settleAnchorAfterFonts(hydratedHeading);
}));
partLinks.forEach(link => link.addEventListener('click', event => {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  const chapter = lastChapterByPart.get(link.dataset.partLink) || (firstChapterByPart.get(link.dataset.partLink) || 1);
  const url = new URL(location.href); url.searchParams.set('chapter', chapter); url.hash = '';
  navigate(url);
}));
document.querySelectorAll('.nav-chapter summary').forEach(summary => summary.addEventListener('click', event => {
  if (event.target.closest('a')) return;
  const details = summary.parentElement;
  details.open = !details.open;
  event.preventDefault();
}));
menu.addEventListener('click', () => {
  const open = sidebar.classList.toggle('open');
  menu.setAttribute('aria-expanded', String(open));
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeMenu();
});
document.querySelector('.content').addEventListener('click', () => {
  if (sidebar.classList.contains('open')) closeMenu();
});
bindCopies(document);
document.addEventListener('click', event => {
  if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  const link=event.target.closest('a[href]');
  if (!link) return;
  const url=new URL(link.href,location.href);
  if (url.origin===location.origin && url.pathname===location.pathname && url.searchParams.has('chapter')) {
    event.preventDefault(); navigate(url);
  }
});
search.addEventListener('input', () => {
  if (!results) return;
  const query = search.value.trim().toLowerCase();
  results.replaceChildren();
  results.hidden = !query;
  if (chapterNav) chapterNav.hidden = !!query;
  if (sidebar) sidebar.scrollTop = 0;
  if (!query) return;
  const matches = searchIndex.filter(item => (item.title + '\n' + item.text).toLowerCase().includes(query));
  const count = document.createElement('p');
  count.textContent = matches.length ? matches.length + ' 个章节包含此内容' : '没有匹配内容';
  results.append(count);
  for (const item of matches) {
    const link = document.createElement('a');
    link.className = 'search-result'; link.href = '?chapter=' + item.chapter;
    const title = document.createElement('strong'); title.textContent = item.title;
    const snippet = document.createElement('small');
    const snippetText = item.text.replace(/\[((?:\\.|[^\]])+)\]\((?:<[^>]*>|(?:\\.|[^)])*)\)/g, '$1');
    const matchOffset = snippetText.toLowerCase().indexOf(query);
    const offset = Math.max(0, (matchOffset < 0 ? 0 : matchOffset) - 35);
    snippet.textContent = snippetText.slice(offset, offset + 125).replace(/[#*~]/g, '');
    link.append(title, snippet);
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const url = new URL(location.href); url.searchParams.set('chapter', item.chapter); url.hash = '';
      navigate(url);
    });
    results.append(link);
  }
});
addEventListener('popstate', () => applyState(readLocation()));
addEventListener('hashchange', () => applyState(readLocation()));
addEventListener('scroll', progress, {passive: true});
history.scrollRestoration = 'manual';
applyState(readLocation());
window.courseNavigation = {navigate, readLocation, applyState};
