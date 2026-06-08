# Sharing a Report

A Pixel Patrol report is a single `.parquet` file holding everything about your dataset - metrics, thumbnails, schema, the works. Pair it with a viewer and you get the full interactive experience; this page covers all the ways to get a viewer to your collaborators (or the other way around).

---

## Which method fits you?

<div class="wc-setup" id="sh-setup-panel">
  <div class="wc-setup-title">📤 Tell us about your situation</div>
  <p style="font-size:0.82rem;margin:0 0 0.7rem;opacity:0.8">Answer the three questions below and the cards for methods that don't fit will dim out - so the right one stands out at a glance.</p>
  <div class="wc-setup-row">
    <span class="wc-setup-q">Roughly how big is your report?</span>
    <span class="wc-setup-btns">
      <button class="wc-setup-btn" data-key="size" data-val="small" onclick="shSetup(this)">Under a few GB</button>
      <button class="wc-setup-btn" data-key="size" data-val="large" onclick="shSetup(this)">5 GB or more</button>
    </span>
  </div>
  <div class="wc-setup-row">
    <span class="wc-setup-q">Does your collaborator need to see any custom widgets you've built?</span>
    <span class="wc-setup-btns">
      <button class="wc-setup-btn" data-key="custom" data-val="yes" onclick="shSetup(this)">Yes, I have custom widgets</button>
      <button class="wc-setup-btn" data-key="custom" data-val="no"  onclick="shSetup(this)">No, the built-ins are enough</button>
    </span>
  </div>
  <div class="wc-setup-row">
    <span class="wc-setup-q">Is this for one person (or a small group), or for anyone to find online?</span>
    <span class="wc-setup-btns">
      <button class="wc-setup-btn" data-key="goal" data-val="person"  onclick="shSetup(this)">One person / small group</button>
      <button class="wc-setup-btn" data-key="goal" data-val="publish" onclick="shSetup(this)">Publishing publicly</button>
    </span>
  </div>
</div>

<p style="font-size:0.82rem;margin:0.5rem 0;opacity:0.8">Click the ✓ in the corner of each card to mark it as reviewed and track your progress.</p>

<div class="wc-progress-wrap">
  <div class="wc-progress-label" id="sh-prog-label">0 / 4 methods reviewed</div>
  <div class="wc-progress-bar"><div class="wc-progress-fill" id="sh-prog-fill"></div></div>
</div>

---

<div class="wc" id="sh-dragdrop">
<div class="wc-head">
<span class="wc-icon">🖱️</span>
<span class="wc-name">Drag &amp; drop into the hosted viewer</span>
<button class="wc-check" data-sh="dragdrop" onclick="shCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>The zero-effort option. No installation required on either end - send your collaborator the <code>.parquet</code> file and point them to:</p>

<p style="text-align:center;font-size:1.05rem;margin:0.9rem 0"><strong><a href="https://ida-mdc.github.io/pixel-patrol/viewer/" target="_blank">ida-mdc.github.io/pixel-patrol/viewer/</a></strong></p>

<p>They drag the file into the browser and the full interactive viewer loads. Everything runs locally in their browser - no data is uploaded anywhere.</p>

<div class="wc-flags">
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div><strong>Browser memory limits:</strong> the hosted viewer runs DuckDB in a browser WebAssembly context, capped by available browser memory. Files under 1-2 GB typically work fine; 5 GB+ may fail to load or get sluggish - reach for <code>pixel-patrol view</code> instead.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div><strong>No custom widgets:</strong> the hosted viewer only ships with the built-in widgets - your own viewer plugins aren't bundled in. If your collaborator needs to see those, build and host your own viewer instead (see "Host on a static server" below).</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><strong>Best for:</strong> a quick look together over a call, or sending a small/medium report to one person who doesn't have Pixel Patrol installed.</div></div>
</div>

</div>
</div>

---

<div class="wc" id="sh-view">
<div class="wc-head">
<span class="wc-icon">💻</span>
<span class="wc-name">Run <code>pixel-patrol view</code> locally</span>
<button class="wc-check" data-sh="view" onclick="shCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Your collaborator will need Pixel Patrol installed - a single <code>pip install pixel-patrol</code> away (see the <a href="installation.md">installation tutorial</a>). Once it's there, they can open any <code>.parquet</code> file directly:</p>

```bash
pixel-patrol view report.parquet
```

<p>This starts a local server backed by <strong>native DuckDB</strong> - no browser memory ceiling, fast even for very large reports.</p>

<div class="wc-flags">
<div class="wc-flag wc-flag-green"><span class="fi">✅</span><div><strong>The recommended method for files above a few gigabytes.</strong> No upload, no waiting on browser WASM - just your data and your collaborator's machine.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><strong>Custom widgets:</strong> they'll show up here too, but only if your collaborator also installs your extension package - it isn't bundled into <code>pixel-patrol view</code> the way it is into a self-contained HTML or hosted site (see below).</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><strong>Best for:</strong> labmates, frequent collaborators, or anyone already in the Pixel Patrol ecosystem - and the best option once your report gets big, custom widgets and all.</div></div>
</div>

</div>
</div>

---

<div class="wc" id="sh-html">
<div class="wc-head">
<span class="wc-icon">📄</span>
<span class="wc-name">Build a self-contained HTML viewer</span>
<button class="wc-check" data-sh="html" onclick="shCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Package the viewer into a single HTML file that works with any parquet:</p>

```bash
pixel-patrol build-viewer-html -o viewer.html
```

<p>Send both <code>viewer.html</code> and <code>report.parquet</code> to your collaborator. They open <code>viewer.html</code> in any browser, pick the parquet from the file picker, and get the full interactive experience - with nothing to install and no external link to track down.</p>

<details class="wc-how">
<summary>🔬 What's actually inside <code>viewer.html</code></summary>
<div><code>viewer.html</code> is the application shell, not the data - your collaborator still needs the <code>.parquet</code> file alongside it (same folder is easiest; the viewer offers a file picker on load). Its core - DuckDB, Plotly, your extensions - is bundled directly into the file. Its styling (Bootstrap CSS and icons), however, still loads from a CDN at runtime, so your collaborator needs an internet connection the <em>first</em> time they open it, even though nothing needs installing.</div>
</details>

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><strong>Best for:</strong> handing someone a self-sufficient package - by email, drive, or USB stick - without needing them to find or trust an external URL.</div></div>
</div>

</div>
</div>

---

<div class="wc" id="sh-static">
<div class="wc-head">
<span class="wc-icon">🌐</span>
<span class="wc-name">Host on a static server</span>
<button class="wc-check" data-sh="static" onclick="shCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Want to publish results online - on GitHub Pages, an institutional server, an S3 bucket, or any static host? Deploy the viewer as a site folder and link straight to your parquet.</p>

<div class="wc-flags">
<div class="wc-flag wc-flag-green"><span class="fi">✅</span><div>This is also the natural way to share a <strong>custom viewer</strong>: any extensions installed in your environment - including your own <a href="extension.md">viewer plugins</a> - are discovered and bundled into the site folder automatically, so your collaborators get your custom widgets without installing anything.</div></div>
</div>

<p><strong>Step 1 - build a viewer site folder:</strong></p>

```bash
pixel-patrol build-viewer-html -o my-report-site/
```

<p>This creates a <code>my-report-site/</code> directory with <code>index.html</code> and all viewer assets.</p>

<p><strong>Step 2 - upload your parquet alongside the site</strong> (or to any public URL).</p>

<p><strong>Step 3 - link to the viewer with a <code>?data=</code> parameter:</strong></p>

```
https://your-host.com/my-report-site/?data=https://your-host.com/report.parquet
```

<p>The viewer fetches the parquet from the URL, so it can live anywhere publicly reachable - S3, GitHub releases, institutional storage, you name it.</p>

<details class="wc-how">
<summary>🔬 GitHub Pages, end to end</summary>
<div>

```bash
# Build the viewer into your gh-pages output
pixel-patrol build-viewer-html -o docs/viewer/

# Add the parquet to the same repo (or reference an external URL)
cp report.parquet docs/viewer/

# After deploying to GitHub Pages, share:
# https://your-org.github.io/your-repo/viewer/?data=https://your-org.github.io/your-repo/viewer/report.parquet
```

</div>
</details>

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><strong>Skip the deploy entirely:</strong> if your parquet is already publicly accessible, the hosted Pixel Patrol viewer accepts a <code>?data=</code> URL too - <code>https://ida-mdc.github.io/pixel-patrol/viewer/?data=https://your-server.com/report.parquet</code>. Anyone with that link opens the full interactive report immediately, no deployment required.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><strong>Best for:</strong> publishing results alongside a paper, a project page, or a team dashboard that anyone can land on.</div></div>
</div>

</div>
</div>

---

## Bonus: share exactly what matters

<div class="wc-ctrl-grid" style="grid-template-columns:repeat(auto-fit,minmax(320px,1fr))">

<div class="wc-ctrl">
<div class="wc-ctrl-head"><span>✂️</span> Share a filtered subset</div>
<div class="wc-ctrl-body">After exploring your report, <strong>apply your filters</strong> in the sidebar (by quality metric, condition, file type, ...), then <strong>Save as Parquet</strong> to export the current view as a new, fully-interactive report - or <strong>Save as CSV</strong> for the same data as a plain spreadsheet, handy for building include/exclude lists or just browsing the numbers yourself. Share either via any method above. Useful for sending a "clean" dataset after removing outliers, handing a teammate just their relevant subset, or publishing a curated, low-noise version of your results.</div>
</div>

<div class="wc-ctrl">
<div class="wc-ctrl-head"><span>🏷️</span> Name it before you share it</div>
<div class="wc-ctrl-body">If you're processing specifically to share, set <code>--name</code> and <code>--description</code> so recipients aren't left guessing what they're looking at - both appear at the top of every viewer session:<br><br>

```bash
pixel-patrol process /data/my_experiment \
    -o experiment_report.parquet \
    --name "Experiment 42 - Drug Screen" \
    --description "Round 1, n=384, 20x brightfield. Processed 2026-06-07."
```
</div>
</div>

</div>

---

<script>
/* ── Setup toggle ─────────────────────────────────────────── */
const shState = { size: null, custom: null, goal: null };

function shSetup(btn) {
  const key = btn.dataset.key;
  const val = btn.dataset.val;
  shState[key] = val;
  document.querySelectorAll('.wc-setup-btn[data-key="' + key + '"]').forEach(b => {
    b.classList.toggle('sel', b.dataset.val === val);
  });
  shApply();
}

function shApply() {
  const dim = (id, unavailable) => {
    const card = document.getElementById(id);
    if (card) card.classList.toggle('wc-unavailable', unavailable);
  };
  dim('sh-dragdrop', shState.size === 'large' || shState.custom === 'yes' || shState.goal === 'publish');
  dim('sh-view',     shState.goal === 'publish');
  dim('sh-html',     shState.size === 'large' || shState.goal === 'publish');
  dim('sh-static',   shState.goal === 'person');
}

/* ── Progress tracker ─────────────────────────────────────── */
const SH_TOTAL = 4;
const SH_KEY   = 'pp-sharing-progress';

function shCheck(btn) {
  const id = btn.dataset.sh;
  const on = btn.classList.toggle('checked');
  btn.textContent = on ? '✓' : '';
  const saved = JSON.parse(localStorage.getItem(SH_KEY) || '{}');
  saved[id] = on;
  localStorage.setItem(SH_KEY, JSON.stringify(saved));
  shUpdateProgress();
}

function shUpdateProgress() {
  const saved = JSON.parse(localStorage.getItem(SH_KEY) || '{}');
  const done  = Object.values(saved).filter(Boolean).length;
  const pct   = Math.round(done / SH_TOTAL * 100);
  const fill  = document.getElementById('sh-prog-fill');
  const label = document.getElementById('sh-prog-label');
  if (fill)  fill.style.width  = pct + '%';
  if (label) label.textContent = done + ' / ' + SH_TOTAL + ' methods reviewed';
}

function shInit() {
  const saved = JSON.parse(localStorage.getItem(SH_KEY) || '{}');
  document.querySelectorAll('.wc-check').forEach(btn => {
    if (saved[btn.dataset.sh]) {
      btn.classList.add('checked');
      btn.textContent = '✓';
    }
  });
  shUpdateProgress();
}

document.addEventListener('DOMContentLoaded', shInit);
</script>
