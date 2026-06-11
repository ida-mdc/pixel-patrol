# Exploring the Report

Open your report in one browser tab and keep this page in another. No report of your own yet? <a href="https://ida-mdc.github.io/pixel-patrol/viewer/?data=../example.parquet&group=imported_path_short" target="_blank">Open the example dataset</a> used for the screenshots below and follow along with that instead.

The example screenshots below are from the WHOI plankton microscopy dataset, which we deliberately tampered with to create four synthetic conditions: **original** images, **blurred** images, **JPEG-compressed** images, and **noisy** images. This makes it easy to see what each widget catches.

A few things to know before you start:

- **Some charts collapse into a table.** If all your images share the same value for a property (e.g. they're all the same size, or all the same dtype), Pixel Patrol shows a summary table instead of a chart.
- **Widget availability depends on what was processed.** Widgets that need data not collected simply don't appear. The colored pill on each card tells you what's required.
- **Everything reacts to the sidebar.** Grouping, filters, and dimension selectors update all widgets instantly - we'll cover the sidebar [at the end](#working-with-the-sidebar).

**Reading the cards:**

<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.75rem 0 1.5rem;font-size:0.82rem;align-items:center">
  <span class="wc-pill wc-pill-always">always shown</span> &nbsp;visible in every report &nbsp;&nbsp;
  <span class="wc-pill wc-pill-loader">needs loader</span> &nbsp;requires an image loader (e.g. <code>pixel-patrol-loader-bio</code>) &nbsp;&nbsp;
  <span class="wc-pill wc-pill-image">needs pixel-patrol-image</span> &nbsp;quality metrics package &nbsp;&nbsp;
  <span class="wc-pill wc-pill-dim">multidim only</span> &nbsp;non-spatial dimensions (e.g. Z/T/C/S) with &gt;1 slice
</div>

Each widget also carries a small **scope badge** in its header, telling you what one datapoint represents: 📄 **per file**, 🖼️ **per image** (one image's worth of data, even if its file holds several), or 🧩 **per slice** (a single channel/Z-plane/timepoint within an image). Hover the badge for details.

---

## Opening your report

```bash
pixel-patrol view report.parquet
```

This starts a local server with native DuckDB and opens the viewer in your browser - recommended for any non-trivial dataset. You can also go to [ida-mdc.github.io/pixel-patrol/viewer/](https://ida-mdc.github.io/pixel-patrol/viewer/) and drag your own `.parquet` in directly (browser-WASM, practical limit ~2 GB).

**Useful launch flags:**

```bash
pixel-patrol view report.parquet --group-by file_extension   # start grouped differently
pixel-patrol view report.parquet --filter-col dtype --filter-op eq --filter-val uint16
pixel-patrol view report.parquet --dim z=5                   # lock to a Z slice on load
pixel-patrol view report.parquet --significance              # show stat brackets from the start
```

---

## Widget walkthrough

<div class="wc-setup" id="wc-setup-panel">
  <div class="wc-setup-title">⚙ What's in your report?</div>
  <p style="font-size:0.82rem;margin:0 0 0.7rem;opacity:0.8">Answer the three questions below and the cards for widgets that don't apply to your setup will dim out as you scroll through the walkthrough - so you can see at a glance which ones are actually relevant to you.</p>
  <div class="wc-setup-row">
    <span class="wc-setup-q">Did you run processing with an image loader (e.g. <code>pixel-patrol-loader-bio</code>)?</span>
    <span class="wc-setup-btns">
      <button class="wc-setup-btn" data-key="loader" data-val="yes" onclick="wcSetup(this)">Yes</button>
      <button class="wc-setup-btn" data-key="loader" data-val="no"  onclick="wcSetup(this)">No</button>
    </span>
  </div>
  <div class="wc-setup-row">
    <span class="wc-setup-q">Did you have <code>pixel-patrol-image</code> installed at processing time?</span>
    <span class="wc-setup-btns">
      <button class="wc-setup-btn" data-key="image" data-val="yes" onclick="wcSetup(this)">Yes</button>
      <button class="wc-setup-btn" data-key="image" data-val="no"  onclick="wcSetup(this)">No</button>
    </span>
  </div>
  <div class="wc-setup-row">
    <span class="wc-setup-q">Are your images multidimensional (e.g. Z, T, C, or S axes)?</span>
    <span class="wc-setup-btns">
      <button class="wc-setup-btn" data-key="dim" data-val="yes" onclick="wcSetup(this)">Yes</button>
      <button class="wc-setup-btn" data-key="dim" data-val="no"  onclick="wcSetup(this)">No</button>
    </span>
  </div>
</div>

<p style="font-size:0.82rem;margin:0.5rem 0;opacity:0.8">Click the ✓ in the corner of each widget card to mark it as reviewed and track your progress through the walkthrough.</p>

<div class="wc-progress-wrap">
  <div class="wc-progress-label" id="wc-prog-label">0 / 13 widgets reviewed</div>
  <div class="wc-progress-bar"><div class="wc-progress-fill" id="wc-prog-fill"></div></div>
</div>

---

<div class="wc" data-wc-req="" id="wc-summary">
<div class="wc-head">
<span class="wc-icon">📋</span>
<span class="wc-name">File Data Summary</span>
<span class="wc-pill wc-pill-always">always shown</span>
<button class="wc-check" data-wc="summary" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Your first sanity check: how many files are in each group, their total size on disk, and which file formats are present.</p>

<div class="wc-shots two-col">
  <div class="wc-shot">
    <img src="../../assets/screenshots/summ1.png" alt="File Count per Group">
    <figcaption>File count per group - balanced at 10 each.</figcaption>
  </div>
  <div class="wc-shot">
    <img src="../../assets/screenshots/sum2.png" alt="Total Size per Group">
    <figcaption>Total size per group - <code>condition3_comp</code> is 10× smaller. Already suspicious before looking at a single pixel.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div>Uneven group sizes can skew statistics and significance tests.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>One condition being much smaller on disk than the others (same file count, much less data) is a hint of a possible issue - images that differ in size or dtype, or compression. The next few widgets will help you tell which it is.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="" id="wc-image-table">
<div class="wc-head">
<span class="wc-icon">📋</span>
<span class="wc-name">Image Table</span>
<span class="wc-pill wc-pill-always">always shown</span>
<button class="wc-check" data-wc="image-table" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>A sortable, searchable table with one row per image, showing every column except thumbnails and other binary/array data. Click a column header to sort; press Enter in the search box to search by substring - across all columns for datasets under 10,000 images, or just <code>path</code>/<code>child_id</code> for larger ones.</p>

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>The fastest way to find a specific file or check raw values before trusting a chart - sort by a metric to see the exact numbers behind the most extreme points in the violin plots.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>Use <strong>Save as CSV</strong> in the sidebar if you want the full table (filtered or not) outside the browser.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="" id="wc-file-stats">
<div class="wc-head">
<span class="wc-icon">📁</span>
<span class="wc-name">File Statistics</span>
<span class="wc-pill wc-pill-always">always shown</span>
<button class="wc-check" data-wc="file-stats" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>File count and total size by extension, file size distribution, and a modification timeline.</p>

<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>File metadata only - no pixel data. Sizes from the filesystem; modification timestamps from each file's <code>mtime</code>.</div>
</details>

<div class="wc-shots two-col">
  <div class="wc-shot">
    <img src="../../assets/screenshots/ext.png" alt="File Count by Extension">
    <figcaption>File count by extension - <code>condition3_comp</code> is the only group with <code>.jpeg</code> files. JPEG compression probably explains the smaller file sizes.</figcaption>
  </div>
  <div class="wc-shot">
    <img src="../../assets/screenshots/count_ext.png" alt="File Count by Size Bin">
    <figcaption>File size distribution - the compressed JPEG images cluster in the small-size bins on the left; originals and noisy images spread to the right.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div><strong>One condition has a different file format than the others.</strong> In this example, only <code>condition3_comp</code> has <code>.jpeg</code> files. Mixed extensions across groups can mean a mixed dataset - images exported in two formats, or even saved twice.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>Modification dates spread across weeks or months - e.g. one condition acquired in March and another in May - you may want to check for a batch effect.</div></div>
<div class="wc-flag wc-flag-green"><span class="fi">✅</span><div>Most properties collapsed to tables and modification dates are clustered together - in many cases, that points to consistent acquisition.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="" id="wc-sunburst">
<div class="wc-head">
<span class="wc-icon">🌀</span>
<span class="wc-name">File System Structure</span>
<span class="wc-pill wc-pill-always">always shown</span>
<button class="wc-check" data-wc="sunburst" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>An interactive sunburst of your folder hierarchy. A toggle lets you size the segments by <strong>file count</strong> or by <strong>total size</strong> - the screenshot below shows the by-size view. Only files that were actually scanned into the report are included. <strong>Click any segment to zoom in; click the center ring to zoom out.</strong></p>

<div class="wc-shots">
  <div class="wc-shot">
    <img src="../../assets/screenshots/burst.png" alt="File Structure Sunburst">
    <figcaption>The 4 conditions at the second ring, each split into species subfolders - each file appears at the outer edge. Hover any segment to see the path and count.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>Use this to verify your dataset is structured as expected.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="loader" id="wc-metadata">
<div class="wc-head">
<span class="wc-icon">🧬</span>
<span class="wc-name">Metadata</span>
<span class="wc-pill wc-pill-loader">needs loader</span>
<button class="wc-check" data-wc="metadata" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Distribution of <code>dtype</code> and <code>dim_order</code> across groups, plus a summary table of other properties - <code>ndim</code> and pixel size, where available - for the cases where every file shares the same value.</p>

<div class="wc-flags">
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div><strong>If <code>dtype</code>, <code>dim_order</code>, or <code>ndim</code> vary across your files</strong>, that points to inconsistent acquisition - possibly different instruments or different acquisition parameters - and is worth verifying before you compare groups. Mixed-<code>dtype</code> example: <code>uint8</code> lives in [0, 255] while <code>uint16</code> lives in [0, 65535], so a pixel value of 100 means something completely different in each.</div></div>
<div class="wc-flag wc-flag-green"><span class="fi">✅</span><div>Single <code>dtype</code>, <code>dim_order</code>, and <code>ndim</code> across the whole dataset - format-level consistency confirmed.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="loader" id="wc-dim-size">
<div class="wc-head">
<span class="wc-icon">📐</span>
<span class="wc-name">Dimension Size Distribution</span>
<span class="wc-pill wc-pill-loader">needs loader</span>
<button class="wc-check" data-wc="dim-size" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>X/Y scatter plot (width vs. height, one point per image) plus strip plots for any non-spatial dimensions present (e.g. Z, T, C, S - names depend on your loader). Quickly reveals outliers and whether your images are consistently sized across groups.</p>

<div class="wc-shots two-col match-height">
  <div class="wc-shot">
    <img src="../../assets/screenshots/yx.png" alt="X vs Y size scatter">
    <figcaption>X vs Y size - plankton images vary widely (some organisms are much larger). No tight cluster here is expected for a species-diverse dataset.</figcaption>
  </div>
  <div class="wc-shot">
    <img src="../../assets/screenshots/y_size.png" alt="Y size distribution">
    <figcaption>Y size violin - <code>condition1_org</code> has the widest spread, including one very tall image (~580px). That outlier is worth investigating.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div>Two distinct clusters in the X/Y scatter - e.g. a 1024×1024 group and a separate 512×512 group - often indicates images from different instruments, different acquisition parameters, or accidentally included thumbnails.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>Hover over any point to see the file path. A single size outlier is often a misplaced metadata file or a preview image saved alongside real data.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="loader" id="wc-mosaic">
<div class="wc-head">
<span class="wc-icon">🖼️</span>
<span class="wc-name">Image Mosaic</span>
<span class="wc-pill wc-pill-loader">needs loader</span>
<button class="wc-check" data-wc="mosaic" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>A thumbnail grid - one patch per image, border color = group. Sort by any metric and your statistics become something you can actually look at and verify with your own eyes.</p>

<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>The thumbnail processor extracts a 64×64-pixel patch per image. For large images it samples across the full spatial extent. For Z-stacks it takes the middle slice. For multi-channel images it picks the most informative channel (RGB if present, otherwise the first). Each patch is independently intensity-normalized so details are visible regardless of absolute brightness.</div>
</details>

<div class="wc-shots">
  <div class="wc-shot">
    <img src="../../assets/screenshots/mos.png" alt="Image Mosaic">
    <figcaption>Plankton thumbnails from all four conditions - border colors indicate group. Sort by <code>laplacian_variance</code> ascending and the blurred/out-of-focus images will float to the top.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><strong>Sort by <code>laplacian_variance</code> ascending</strong> → blurriest images appear first. If they look visually out-of-focus, they are. This is the fastest path to finding focus-drift candidates for exclusion.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>Sort by <code>blocking_index</code> descending to surface JPEG-artifact images. Sort by <code>max_intensity</code> descending to surface potentially saturated ones.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>Hover over any thumbnail to see the exact file path.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="loader" id="wc-violin-basic">
<div class="wc-head">
<span class="wc-icon">📈</span>
<span class="wc-name">Pixel Value Statistics</span>
<span class="wc-pill wc-pill-loader">needs loader</span>
<button class="wc-check" data-wc="violin-basic" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Violin + box plots for four per-image statistics: <code>mean_intensity</code>, <code>std_intensity</code>, <code>min_intensity</code>, <code>max_intensity</code>. Each dot is one image; each violin is one group.</p>

<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>
<strong>mean:</strong> <code>np.nanmean</code> over all pixels, pixel-count-weighted across 2D planes.<br>
<strong>std:</strong> pooled - <code>√(Σnᵢ(σᵢ² + (μᵢ − μ̄)²) / Σnᵢ)</code> - accounts for both within-plane and between-plane variance.<br>
<strong>min / max:</strong> <code>np.nanmin</code> / <code>np.nanmax</code> per plane, then min/max across planes.<br>
NaN pixels are excluded from all calculations.
</div>
</details>

<div class="wc-shots">
  <div class="wc-shot">
    <img src="../../assets/screenshots/max_intensity.png" alt="Max Intensity violin">
    <figcaption><code>max_intensity</code> distribution - <code>condition2_bl</code> (blurred) has a wide, low distribution. <code>condition1_org</code> clusters tightly near 255. Each dot is one image; hover to identify the file.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div><strong><code>max_intensity</code> sitting exactly at 255 (or 65535)</strong> is worth a closer look - it can mean those images hit the sensor's ceiling and got clipped. Check the histogram for a spike at the max value to confirm real saturation (a single bright pixel can also legitimately land at 255), then verify visually in the mosaic.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>Large spread in <code>mean_intensity</code> within a single group: variable illumination, inconsistent staining, or genuinely variable signal.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>Enable <strong>Show significance</strong> in the sidebar for Mann-Whitney U brackets (Bonferroni corrected). A bracket = statistically distinguishable groups - whether that's meaningful is your call.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="loader" id="wc-histogram">
<div class="wc-head">
<span class="wc-icon">📊</span>
<span class="wc-name">Pixel Value Histograms</span>
<span class="wc-pill wc-pill-loader">needs loader</span>
<button class="wc-check" data-wc="histogram" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>The mean pixel intensity histogram per group - a more in-depth look at whether your pixel intensity distribution looks the way you'd expect. Toggle between a normalized <strong>0-255</strong> view and the <strong>native range</strong> of your data.</p>

<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>Each image gets a 256-bin histogram of its pixel values, normalized to sum to 1. Per-group, histograms are averaged using pixel-count weighting - larger images contribute proportionally more.</div>
</details>

<div class="wc-shots">
  <div class="wc-shot">
    <img src="../../assets/screenshots/hist.png" alt="Intensity histograms">
    <figcaption>The four conditions have strikingly different distributions: <code>condition2_bl</code> (blurred, orange) peaks around 190 - much brighter on average. <code>condition3_comp</code> (JPEG, cyan) is spread unusually flat. All four peak at 0 (dark background), but to very different degrees.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>Comparing histograms across groups side by side is a quick way to look at conditions in more depth than a single summary statistic can show.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>Distributions shifted between groups: could be real biology, or different acquisition settings on different days.</div></div>
<div class="wc-flag wc-flag-green"><span class="fi">✅</span><div>Distribution looks the way you'd expect for your sample, and is consistent across groups that should be comparable. (Not every dataset should be bell-shaped, and not every condition should look alike - judge against your own expectations, not a generic template.)</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="loader,image" id="wc-violin-quality">
<div class="wc-head">
<span class="wc-icon">🎯</span>
<span class="wc-name">Image Quality Metrics</span>
<span class="wc-pill wc-pill-loader">needs loader</span>
<span class="wc-pill wc-pill-image" style="margin-left:4px">needs pixel-patrol-image</span>
<button class="wc-check" data-wc="violin-quality" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Five quality metrics computed per leaf slice, aggregated by pixel-count-weighted mean. Together they surface a wide range of issues - focus problems, low contrast, sensor noise, uneven texture, and compression artifacts - not just one of these. <strong>Only compare images at the same magnification and pixel size</strong> - all these metrics are scale-dependent.</p>

<div class="wc-metric">
<div class="wc-metric-name">Laplacian Variance <small style="font-weight:400;opacity:0.65">- sharpness / focus quality</small></div>
<p style="font-size:0.9rem;margin:0 0 0.5rem">The go-to blur detector. High = sharp; low = blurry or out-of-focus.</p>
<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>Apply the discrete Laplacian at every pixel: <code>left + right + up + down − 4 × center</code>. This second-derivative filter fires strongly at edges and fine texture, near-zero in smooth regions. Blurry image → small response → low variance. Sharp image → large response → high variance.</div>
</details>
<div class="wc-flags">
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div>Low outliers in the violin → go to the mosaic, sort by <code>laplacian_variance</code> ascending, and confirm visually. If they look out-of-focus, maybe you want to exclude them.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>One condition consistently blurrier: different imaging days, focus protocols, or objective settings?</div></div>
</div>
</div>

<div class="wc-metric">
<div class="wc-metric-name">Michelson Contrast <small style="font-weight:400;opacity:0.65">- local dynamic range</small></div>
<p style="font-size:0.9rem;margin:0 0 0.5rem">How much local contrast the image has - whether pixel values vary substantially within small neighborhoods.</p>
<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>For every 3×3 window, compute <code>max − min</code> (local intensity range). Average all local ranges across the image. Divide by the global spatial standard deviation. Higher = more local contrast relative to overall spread.</div>
</details>
<div class="wc-flags">
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>Very low Michelson contrast: image is "flat" - possible underexposure, oversaturation, or genuinely low-contrast sample. Compare with the histogram to distinguish.</div></div>
</div>
</div>

<div class="wc-metric">
<div class="wc-metric-name">MSCN Variance <small style="font-weight:400;opacity:0.65">- noise sensitivity (BRISQUE)</small></div>
<p style="font-size:0.9rem;margin:0 0 0.5rem">A no-reference quality metric sensitive to both noise and blur.</p>
<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>Normalize each pixel by subtracting its 3×3 local mean and dividing by local std (+ small stabilizer). In a well-structured image this normalized map approaches zero everywhere. The variance of that map measures residual complexity: noisy → high; blurry → low.</div>
</details>
<div class="wc-flags">
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>Abnormally high: pixel-level noise - possibly high sensor gain or short exposure.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>Very low: smooth and featureless - blur, low signal, or a genuinely uniform sample.</div></div>
</div>
</div>

<div class="wc-metric">
<div class="wc-metric-name">Texture Heterogeneity <small style="font-weight:400;opacity:0.65">- spatial uniformity of texture</small></div>
<p style="font-size:0.9rem;margin:0 0 0.5rem">How unevenly texture is distributed - high means some regions are rich while others are flat.</p>
<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>Compute local std in every 3×3 window. Then compute the coefficient of variation (std / mean) of all those local stds. High = texture concentrated in patches; low = texture uniformly distributed.</div>
</details>
<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>High heterogeneity is expected for sparse fluorescence images (textured cells on flat background). Interesting only when it differs unexpectedly between conditions.</div></div>
</div>
</div>

<div class="wc-metric">
<div class="wc-metric-name">Blocking Index <small style="font-weight:400;opacity:0.65">- JPEG compression artifacts</small></div>
<p style="font-size:0.9rem;margin:0 0 0.5rem">Detects 8×8-pixel block boundaries - the hallmark signature of JPEG compression.</p>
<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>Measure the average absolute pixel jump across every 8-pixel grid boundary (horizontal and vertical). JPEG encodes in 8×8 blocks, creating subtle discontinuities at those boundaries.</div>
</details>

<div class="wc-shots">
  <div class="wc-shot">
    <img src="../../assets/screenshots/blocking.png" alt="Blocking Index violin">
    <figcaption><code>condition3_comp</code> (JPEG) has the highest blocking index - confirming what the file extension chart already hinted at. <code>condition2_bl</code> (blurred) has the lowest, because blurring smooths the block edges.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div>Non-zero blocking in data that should be lossless (TIFF, ND2, CZI): the data likely went through lossy compression somewhere - an export, upload, or storage step. Lossy compression should generally be avoided for scientific data: it can't be undone, and it distorts quantitative measurements.</div></div>
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div>High blocking in one condition but not another: they were processed differently. A systematic artifact that will skew any comparison between them.</div></div>
</div>
</div>

<div class="wc-metric">
<div class="wc-metric-name">Ringing Index <small style="font-weight:400;opacity:0.65">- edge oscillation artifacts</small></div>
<p style="font-size:0.9rem;margin:0 0 0.5rem">High-frequency oscillations near edges - ringing from lossy compression or aggressive filtering.</p>
<details class="wc-how">
<summary>🔬 How it's computed</summary>
<div>Subtract a 3×3 box-average from each pixel (high-pass filter). Compute variance of the residual. High variance = fine oscillations on top of the main image structure.</div>
</details>
<div class="wc-flags">
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>High ringing + high blocking = strong evidence of lossy compression. High ringing alone can also come from aggressive spatial filtering (e.g. unsharp masking) applied during acquisition or export.</div></div>
</div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="dim" id="wc-stats-basic">
<div class="wc-head">
<span class="wc-icon">📉</span>
<span class="wc-name">Basic Statistics Across Dimensions</span>
<span class="wc-pill wc-pill-dim">multidim only</span>
<button class="wc-check" data-wc="stats-basic" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>How <code>mean</code>, <code>std</code>, <code>min</code>, and <code>max</code> change as you move along your non-spatial dimensions (e.g. Z, T, C, or S - the exact names depend on your loader) - averaged across all images in each group.</p>

<div class="wc-shots compact">
  <div class="wc-shot">
    <img src="../../assets/screenshots/max_per_s_slice.png" alt="Max intensity per S slice">
    <figcaption>Max intensity across S (channel) slices - different images peak at different channels, and the spread widens at slice 2. The shaded band is the per-group std across images.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div><code>mean_intensity</code> declining over T: <strong>photobleaching</strong>. Fluorophores are bleaching over the time course - any time-series analysis must account for this.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div><code>std_intensity</code> increasing over T: growing noise - sample movement, degradation, or instrument drift.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>A channel with very low mean and high relative std likely has no real signal - just noise. Confirm visually in the mosaic for that channel.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>Each plot also has a dashed line - the percentage of images that still have a slice at that position, per group. A line dropping below 100% means images in that group don't all have the same number of slices.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="loader,image,dim" id="wc-stats-quality">
<div class="wc-head">
<span class="wc-icon">🎨</span>
<span class="wc-name">Quality Metrics Across Dimensions</span>
<span class="wc-pill wc-pill-loader">needs loader</span>
<span class="wc-pill wc-pill-image" style="margin-left:4px">needs pixel-patrol-image</span>
<span class="wc-pill wc-pill-dim" style="margin-left:4px">multidim only</span>
<button class="wc-check" data-wc="stats-quality" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>How quality metrics change across your non-spatial dimensions (e.g. Z, T, C, or S - names depend on your loader). Best for detecting focus drift across a Z-like dimension or photobleaching effects on contrast over a T-like one.</p>

<div class="wc-shots compact">
  <div class="wc-shot">
    <img src="../../assets/screenshots/blocking_per_s_slice.png" alt="Blocking index per S slice">
    <figcaption>Blocking index across S (channel) slices - <code>condition4_nois</code> (noisy, dark green) increases sharply at slice 2, while <code>condition2_bl</code> (blurred, orange) stays near zero across all slices.</figcaption>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-red"><span class="fi">🚩</span><div><strong>Laplacian variance dropping across Z:</strong> focus is unstable across the stack. Find the Z range where sharpness peaks and limit your analysis to that range.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>Michelson contrast dropping over T: signal-to-background shrinking - consistent with photobleaching.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>MSCN variance spiking at one specific T frame: sample motion, a bubble, or a transient autofluorescence event.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>Find the Z of peak Laplacian variance here, then set that Z in the sidebar dimension selector - all widgets update to show your data at its sharpest.</div></div>
<div class="wc-flag wc-flag-yellow"><span class="fi">⚠️</span><div>As with Basic Statistics Across Dimensions, the dashed line shows the per-group percentage of images that still have a slice at that position - check it before reading too much into a quality metric at the far end of a dimension.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-wc-req="" id="wc-custom-plot">
<div class="wc-head">
<span class="wc-icon">🧪</span>
<span class="wc-name">Custom Plot</span>
<span class="wc-pill wc-pill-always">always shown</span>
<button class="wc-check" data-wc="custom-plot" onclick="wcCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Build your own plot from any columns in the report - pick an X column, a Y column (or <code>(count)</code>), and Pixel Patrol picks a sensible chart type:</p>

<ul style="font-size:0.9rem;line-height:1.6">
<li><strong>Two numerics</strong> → scatter</li>
<li><strong>Categorical × numeric</strong> → violin or bar (mean ± sd)</li>
<li><strong>Any column × <code>(count)</code></strong> → count bar</li>
<li><strong>Two categoricals</strong> → count heatmap</li>
</ul>

<p><strong>Color by</strong> defaults to the app-wide group column, but you can color/split by any other column instead - or, for scatter plots, by a numeric column on a continuous colormap. Click <strong>＋ Add plot</strong> for as many independent plots as you need.</p>

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>Use this for anything not covered by the built-in widgets - e.g. plotting a loader-specific metadata column against a quality metric, or checking whether two metrics correlate.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><strong>↓ Export plugin</strong> downloads the current plot as a standalone viewer plugin <code>.js</code> file - drop it into an extension's <code>viewer/</code> folder and list it in <code>extension.json</code> to make it a permanent part of your reports. See <a href="../extension/">Create an Extension</a>.</div></div>
</div>

</div>
</div>

---

## Working with the sidebar

<div class="wc-ctrl-grid">

<div class="wc-ctrl">
<div class="wc-ctrl-head"><span>🗂️</span> Group by</div>
<div class="wc-ctrl-body">Split images into groups by any column. Default: <code>path</code> (your <code>-p</code> conditions). Try <code>file_extension</code>, <code>dtype</code>, or any loader metadata column to ask different questions.</div>
</div>

<div class="wc-ctrl">
<div class="wc-ctrl-head"><span>🔍</span> Filter</div>
<div class="wc-ctrl-body">Restrict all widgets to rows matching a column + operator + value. Filters stack with grouping - filter to a dtype, then group by path to compare conditions within it.</div>
</div>

<div class="wc-ctrl">
<div class="wc-ctrl-head"><span>📏</span> Dimension selectors</div>
<div class="wc-ctrl-body">For multidimensional data: a slider per non-spatial dimension (e.g. Z, T, C, S - names depend on your loader) that sets which slice all widgets display. Set the Z-like dimension to its peak Laplacian variance for the sharpest cross-condition comparison.</div>
</div>

<div class="wc-ctrl">
<div class="wc-ctrl-head"><span>✨</span> Show significance</div>
<div class="wc-ctrl-body">Mann-Whitney U test brackets on violin plots, Bonferroni corrected. Non-parametric - no normality assumption. A bracket = statistically distinguishable. Biological meaningfulness is your judgment.</div>
</div>

<div class="wc-ctrl">
<div class="wc-ctrl-head"><span>💾</span> Save</div>
<div class="wc-ctrl-body"><strong>CSV:</strong> the filtered file list with all metrics, in a plain spreadsheet format - use it to pick files to include or exclude in your pipeline, or just to browse and sort the full table yourself.<br><strong>Parquet:</strong> a new, fully-interactive PP report of only the images that passed your filters.</div>
</div>

<div class="wc-ctrl">
<div class="wc-ctrl-head"><span>☰</span> Widget list</div>
<div class="wc-ctrl-body">Collapse or expand individual widgets. Grayed-out = required columns not in this report (processor wasn't run).</div>
</div>

<div class="wc-ctrl">
<div class="wc-ctrl-head"><span>💬</span> Feedback</div>
<div class="wc-ctrl-body">Send feedback directly from the viewer. If something looks wrong, a metric is confusing, or you'd like a new feature - tell us here. All feedback is read.</div>
</div>

</div>

**Common filter examples:**

| Goal | Column | Op | Value |
|---|---|---|---|
| Only 16-bit images | `dtype` | `eq` | `uint16` |
| One condition only | `path` | `eq` | `condition_A` |
| Sharp images only | `laplacian_variance` | `gt` | `500` |
| Remove dark images | `mean_intensity` | `gt` | `10` |
| Find saturated images | `max_intensity` | `ge` | `254` |
| Exclude a format | `file_extension` | `not_contains` | `.tif` |

**Saving filtered subsets:** **Save as CSV** gives you the filtered data as a spreadsheet - the same table as the parquet, in a plain, human-readable format. Use it to build an include or exclude list for your pipeline, or just open it and explore the numbers yourself. **Save as Parquet** creates a new fully-interactive report with only the images that passed your filters - the standard way to share a curated, clean version of your dataset report.

---

<script>
/* ── Setup toggle ─────────────────────────────────────────── */
const wcState = { loader: null, image: null, dim: null };

function wcSetup(btn) {
  const key = btn.dataset.key;
  const val = btn.dataset.val;
  wcState[key] = val;
  document.querySelectorAll('.wc-setup-btn[data-key="' + key + '"]').forEach(b => {
    b.classList.toggle('sel', b.dataset.val === val);
  });
  wcApply();
}

function wcApply() {
  document.querySelectorAll('.wc[data-wc-req]').forEach(card => {
    const req = card.dataset.wcReq;
    let unavail = false;
    if (req.includes('loader') && wcState.loader === 'no') unavail = true;
    if (req.includes('image')  && (wcState.image === 'no' || wcState.loader === 'no')) unavail = true;
    if (req.includes('dim')    && wcState.dim    === 'no') unavail = true;
    card.classList.toggle('wc-unavailable', unavail);
  });
}

/* ── Progress tracker ─────────────────────────────────────── */
const WC_TOTAL = 13;
const WC_KEY   = 'pp-report-progress';

function wcCheck(btn) {
  const id = btn.dataset.wc;
  const on = btn.classList.toggle('checked');
  btn.textContent = on ? '✓' : '';
  const saved = JSON.parse(localStorage.getItem(WC_KEY) || '{}');
  saved[id] = on;
  localStorage.setItem(WC_KEY, JSON.stringify(saved));
  wcUpdateProgress();
}

function wcUpdateProgress() {
  const saved = JSON.parse(localStorage.getItem(WC_KEY) || '{}');
  const done  = Object.values(saved).filter(Boolean).length;
  const pct   = Math.round(done / WC_TOTAL * 100);
  const fill  = document.getElementById('wc-prog-fill');
  const label = document.getElementById('wc-prog-label');
  if (fill)  fill.style.width  = pct + '%';
  if (label) label.textContent = done + ' / ' + WC_TOTAL + ' widgets reviewed';
}

function wcInit() {
  const saved = JSON.parse(localStorage.getItem(WC_KEY) || '{}');
  document.querySelectorAll('.wc-check').forEach(btn => {
    if (saved[btn.dataset.wc]) {
      btn.classList.add('checked');
      btn.textContent = '✓';
    }
  });
  wcUpdateProgress();
}

document.addEventListener('DOMContentLoaded', wcInit);
</script>
