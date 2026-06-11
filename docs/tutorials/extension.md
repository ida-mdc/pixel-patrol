# Create an Extension

<img src="../../assets/shark.png" alt="A softly glowing cartoon shark" width="220" align="right">

Pixel Patrol is built to be extended - without forking it. An **extension** is a regular, installable Python package that can add any combination of:

- a custom **loader** - read a file format Pixel Patrol doesn't support out of the box
- a custom **processor** - compute new metrics and add them as report columns
- custom **viewer plugins** - visualize anything in the report with your own widgets

All three are optional, and one package can mix and match freely. The contracts for each (`PixelPatrolLoader`, `PixelPatrolProcessor`, the plugin object shape) are defined as [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)s in `pixel_patrol_base.core.contracts`, not base classes - your classes just need to match the expected shape (the right `NAME`, methods, attributes, ...), with no import or inheritance from `pixel_patrol_base` required. That's what keeps extensions standalone, decoupled packages.

This page walks through all three pieces using **[Pixel HAI Watch](https://github.com/ida-mdc/pixel-patrol/tree/main/examples/minimal-extension)** - a complete, working, slightly playful example bundled with Pixel Patrol. Its twist: there are no real images. `.parquet` *tables* are read as if they were tiny snapshots from a deep-sea shark camera - each table's numeric columns become a pixel grid, and a key/value pair tucked into the file's metadata stands in for the kind of instrument metadata real loaders extract (channel names, pixel sizes, acquisition stamps, ...). Every snippet below is taken directly from it - open `examples/minimal-extension/` alongside this page and follow along.

---

<div class="wc-setup" id="ext-setup-panel">
  <div class="wc-setup-title">⚙ What are you building?</div>
  <p style="font-size:0.82rem;margin:0 0 0.7rem;opacity:0.8">Answer the three questions below and the cards for pieces you don't need will dim out - so you can focus on the ones that matter for your extension.</p>
  <div class="wc-setup-row">
    <span class="wc-setup-q">Do you need to read a file format Pixel Patrol doesn't support yet?</span>
    <span class="wc-setup-btns">
      <button class="wc-setup-btn" data-key="loader" data-val="yes" onclick="extSetup(this)">Yes</button>
      <button class="wc-setup-btn" data-key="loader" data-val="no"  onclick="extSetup(this)">No</button>
    </span>
  </div>
  <div class="wc-setup-row">
    <span class="wc-setup-q">Do you want to compute your own metrics from the image data?</span>
    <span class="wc-setup-btns">
      <button class="wc-setup-btn" data-key="processor" data-val="yes" onclick="extSetup(this)">Yes</button>
      <button class="wc-setup-btn" data-key="processor" data-val="no"  onclick="extSetup(this)">No</button>
    </span>
  </div>
  <div class="wc-setup-row">
    <span class="wc-setup-q">Do you want a custom chart or visualization in the report viewer?</span>
    <span class="wc-setup-btns">
      <button class="wc-setup-btn" data-key="viewer" data-val="yes" onclick="extSetup(this)">Yes</button>
      <button class="wc-setup-btn" data-key="viewer" data-val="no"  onclick="extSetup(this)">No</button>
    </span>
  </div>
</div>

<p style="font-size:0.82rem;margin:0.5rem 0;opacity:0.8">Click the ✓ in the corner of each card to mark it as reviewed and track your progress.</p>

<div class="wc-progress-wrap">
  <div class="wc-progress-label" id="ext-prog-label">0 / 5 pieces reviewed</div>
  <div class="wc-progress-bar"><div class="wc-progress-fill" id="ext-prog-fill"></div></div>
</div>

---

<div class="wc" data-ext-req="" id="ext-anatomy">
<div class="wc-head">
<span class="wc-icon">📦</span>
<span class="wc-name">Anatomy of an Extension</span>
<span class="wc-pill wc-pill-always">always relevant</span>
<button class="wc-check" data-ext="anatomy" onclick="extCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Any extension is a regular, installable Python package. Here's Pixel HAI Watch's layout - the role of each file is the blueprint for your own:</p>

<pre class="wiz-code-pre">pixel_patrol_hai_watch/
├── my_loader.py             custom loader      - reads .parquet "dive patches"
├── my_processor.py          custom processor   - counts the glows in each patch
├── plugin_registry.py       registers loader, processor, and viewer extension
└── viewer/
    ├── extension.json                manifest listing the viewer plugins
    ├── plugin_dives_logged.js        metadata widget    - dives logged, by depth zone &amp; site
    └── plugin_glow_by_depth.js       image-data widget  - glow sightings, by depth zone</pre>

<p>Pixel Patrol finds all of this through Python <strong>entry points</strong>: three optional groups in <code>pyproject.toml</code>, each pointing at a function in your <code>plugin_registry</code> module.</p>

```toml
[project.entry-points."pixel_patrol.loader_plugins"]
my_extension_loaders = "my_package.plugin_registry:register_loader_plugins"

[project.entry-points."pixel_patrol.processor_plugins"]
my_extension_processors = "my_package.plugin_registry:register_processor_plugins"

[project.entry-points."pixel_patrol.viewer_extensions"]
my_extension_viewer = "my_package.plugin_registry:get_viewer_extension_dir"
```

```python
# plugin_registry.py
from pathlib import Path
from my_package.my_loader import MyLoader
from my_package.my_processor import MyProcessor

def register_loader_plugins():
    return [MyLoader]

def register_processor_plugins():
    return [MyProcessor]

def get_viewer_extension_dir():
    return Path(__file__).parent / "viewer"
```

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>You only need to declare the entry-point groups your extension actually uses. A viewer-only extension can omit the loader/processor groups entirely - and a loader-only one can skip the viewer group just as easily.</div></div>
<div class="wc-flag wc-flag-green"><span class="fi">✅</span><div>Once your package is installed in the same environment as <code>pixel_patrol_base</code>, everything is discovered automatically at runtime - no explicit registration or path needed when you call <code>create_project(..., loader="your-loader-name")</code> or <code>serve_viewer(report_path)</code>.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-ext-req="loader" id="ext-loader">
<div class="wc-head">
<span class="wc-icon">📥</span>
<span class="wc-name">The Loader</span>
<button class="wc-check" data-ext="loader" onclick="extCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">🧭</span><div><strong>Build this if</strong> Pixel Patrol can't read your file format - or doesn't read it (and its metadata) the way you want. Maybe your images live in a proprietary instrument format, a database export, or - like here - something delightfully unconventional.</div></div>
</div>

<p>A loader turns a file into a <code>Record</code> - pixel data plus metadata - that the rest of the pipeline can work with. Implement the <code>PixelPatrolLoader</code> protocol:</p>

<table>
<thead><tr><th>Member</th><th>Type</th><th>Required?</th><th>Purpose</th></tr></thead>
<tbody>
<tr><td><code>NAME</code></td><td><code>str</code></td><td>yes</td><td>unique identifier passed to <code>create_project(..., loader=...)</code></td></tr>
<tr><td><code>SUPPORTED_EXTENSIONS</code></td><td><code>set[str]</code></td><td>yes</td><td>file extensions this loader can read (lower-case, no dot)</td></tr>
<tr><td><code>OUTPUT_SCHEMA</code></td><td><code>dict[str, type]</code></td><td>yes</td><td>extra metadata columns this loader adds to the report, with their types</td></tr>
<tr><td><code>read_header(path)</code></td><td><code>(Path) -> FileInfo</code></td><td>yes</td><td>cheap shape/dtype/dim-order probe, <strong>no pixel data loaded</strong></td></tr>
<tr><td><code>load(path)</code></td><td><code>(Path) -> Record</code></td><td>yes</td><td>loads one image and returns a <code>Record</code></td></tr>
<tr><td><code>load_range(path, start, stop)</code></td><td><code>(Path, int, int) -> Iterator[(str, Record)]</code></td><td>yes</td><td>yields sub-images for container formats; raise <code>NotImplementedError</code> otherwise</td></tr>
<tr><td><code>FOLDER_EXTENSIONS</code></td><td><code>set[str]</code></td><td>no</td><td>"extensions" that mark a <em>folder</em> as one loadable unit (e.g. OME-Zarr stores); defaults to empty</td></tr>
<tr><td><code>CONTAINER_EXTENSIONS</code></td><td><code>set[str]</code></td><td>no</td><td>extensions that may contain more than one image (multi-series OME-TIFF, LMDB, ...); defaults to empty</td></tr>
<tr><td><code>OUTPUT_SCHEMA_PATTERNS</code></td><td><code>list[tuple[str, type]]</code></td><td>no</td><td>regex/type pairs for dynamically-named metadata columns (e.g. <code>pixel_size_X</code>); defaults to empty</td></tr>
<tr><td><code>is_folder_supported(path)</code></td><td><code>(Path) -> bool</code></td><td>no</td><td>whether a <em>folder</em> (not a file) should be treated as one image; only relevant if <code>FOLDER_EXTENSIONS</code> is non-empty</td></tr>
</tbody>
</table>

<p><code>SharkCamLoader</code> (<code>NAME = "shark-cam"</code>) reads each table with <code>pyarrow.parquet</code>, stacks its columns into a 2-D array, decodes one field out of the schema metadata, and wraps it all with <code>record_from(...)</code>:</p>

```python
class SharkCamLoader:
    NAME = "shark-cam"

    SUPPORTED_EXTENSIONS = {"parquet"}
    FOLDER_EXTENSIONS    = set()
    CONTAINER_EXTENSIONS = set()

    OUTPUT_SCHEMA          = {"depth_zone": str}
    OUTPUT_SCHEMA_PATTERNS = []

    def is_folder_supported(self, path):
        return False

    def read_header(self, file_path):
        meta = pq.ParquetFile(file_path).metadata
        return FileInfo(shape=(meta.num_rows, meta.num_columns), dtype=np.uint8, dim_order=("Y", "X"))

    def load(self, file_path):
        table = pq.read_table(file_path)

        # Each column is one pixel column (X); stacking them rebuilds the YX grid.
        columns = [table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names]
        pixels = np.column_stack(columns).astype(np.uint8)

        raw_meta = table.schema.metadata or {}
        log_entry = {k.decode(): v.decode() for k, v in raw_meta.items()}
        meta = {
            "depth_zone": log_entry.get("depth_zone", "unknown"),
            "dim_order":  "YX",
        }
        return record_from(pixels, meta, kind="intensity")

    def load_range(self, file_path, start, stop):
        raise NotImplementedError("shark-cam is not a container format")
```

<details class="wc-how">
<summary>🔬 How a parquet table becomes a "dive patch"</summary>
<div>Each file holds a small grid of <code>uint8</code> columns - read column-by-column and stacked side by side, the table <em>is</em> the pixel grid (rows → Y, columns → X). The playful field, <code>depth_zone</code> (sunlit/twilight/midnight/abyss - which layer of the ocean the snapshot was taken in), is decoded straight out of <code>table.schema.metadata</code> - exactly the slot real formats (OME-XML in TIFFs, EXIF in JPEGs, ...) use to carry instrument and acquisition info.</div>
</details>

<div class="wc-flags">
<div class="wc-flag wc-flag-green"><span class="fi">✅</span><div>Declaring <code>kind="intensity"</code> and <code>dim_order="YX"</code> is what makes the <em>built-in</em> processors (basic metrics, histogram, thumbnail) pick the patches up automatically, right alongside your custom one. To a Pixel Patrol pipeline, a "dive patch" parquet table behaves just like any other 2-D image - that's the whole point of the exercise.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><code>read_header</code> is called for every file during the initial scan and must stay cheap - it's your chance to report shape, dtype, and dimension order without paying the cost of loading pixel data.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-ext-req="processor" id="ext-processor">
<div class="wc-head">
<span class="wc-icon">⚙️</span>
<span class="wc-name">The Processor</span>
<button class="wc-check" data-ext="processor" onclick="extCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">🧭</span><div><strong>Build this if</strong> you want to compute a metric on images - any images, regardless of who loaded them. Quality scores, object counts, anything beyond what the built-in processors already cover.</div></div>
</div>

<p>A processor receives loaded records and returns derived values that get merged into the report as new columns. Implement the <code>PixelPatrolProcessor</code> protocol - every member below is required:</p>

<table>
<thead><tr><th>Member</th><th>Type</th><th>Purpose</th></tr></thead>
<tbody>
<tr><td><code>NAME</code></td><td><code>str</code></td><td>unique identifier (shown in pipeline logs)</td></tr>
<tr><td><code>CHUNK_KIND</code></td><td><code>ChunkKind</code></td><td><code>LEAF</code> (user-configured tiles/slices) or <code>MEMORY</code> (whole record, memory-safe chunking)</td></tr>
<tr><td><code>INPUT</code></td><td><code>RecordSpec</code></td><td>which records this processor runs on (<code>kinds</code>, <code>axes</code>, <code>capabilities</code>, ...)</td></tr>
<tr><td><code>OUTPUT</code></td><td><code>"features" | "record"</code></td><td>whether <code>run_chunk</code> returns columns to merge, or a brand-new <code>Record</code></td></tr>
<tr><td><code>OUTPUT_SCHEMA</code></td><td><code>dict[str, type]</code></td><td>the columns this processor adds, with their types</td></tr>
<tr><td><code>run_chunk(record)</code></td><td><code>(Record) -> dict</code></td><td>does the actual computation on one chunk</td></tr>
<tr><td><code>get_aggregation(name)</code></td><td><code>(str) -> Callable | None</code></td><td>how to combine multiple chunks' values for column <code>name</code> into the per-image value</td></tr>
</tbody>
</table>

<p><code>GlowSpotterProcessor</code> (<code>NAME = "glow-spotter"</code>) runs on every <code>intensity</code> record with <code>X</code>/<code>Y</code> axes, uses <code>CHUNK_KIND.LEAF</code>, and adds one column - <code>glow_count</code>:</p>

```python
class GlowSpotterProcessor:
    NAME       = "glow-spotter"
    CHUNK_KIND = ChunkKind.LEAF
    INPUT      = RecordSpec(axes={"X", "Y"}, kinds={"intensity"})
    OUTPUT     = "features"

    OUTPUT_SCHEMA          = {"glow_count": int}
    OUTPUT_SCHEMA_PATTERNS = []

    def run_chunk(self, record):
        arr = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        arr = arr.astype(np.float32)

        threshold = np.median(arr) + 60.0
        glow_count = int(np.sum(arr > threshold))
        return {"glow_count": glow_count}

    def get_aggregation(self, col):
        if col != "glow_count":
            return None
        # Glows are independent per pixel, so chunk counts simply add up.
        return lambda rows, g_dims: sum(r["glow_count"] for r in rows)
```

<details class="wc-how">
<summary>🔬 How "glows" get counted</summary>
<div>A pixel counts as part of a glow when it stands out clearly from the patch's overall brightness - brighter than its median by more than 60. Sunlit patches have almost none; the deeper and darker it gets, the more glows light up - exactly the way real bioluminescence concentrates in the dark, by construction. <code>get_aggregation</code> sums each chunk's <code>glow_count</code> into the per-image total - a pattern that works whenever the thing you're counting is independent per pixel, so splitting an image into pieces and adding the pieces' counts back up reconstructs the whole.</div>
</details>

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><code>CHUNK_KIND</code> shapes how your data arrives, and which unit your computation needs to handle. <code>LEAF</code> - the more common pick for metric processors - tiles large images into memory-safe pieces and runs your computation on each one; <code>MEMORY</code> hands you the whole record at once, which is only safe when you know it comfortably fits in memory.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div><code>OUTPUT = "features"</code> merges your columns into the existing report - the right choice for almost any custom metric. <code>"record"</code> is for processors that produce a brand-new derived image instead (a mask, a projection, ...).</div></div>
</div>

</div>
</div>

---

<div class="wc" data-ext-req="viewer" id="ext-viewer">
<div class="wc-head">
<span class="wc-icon">📊</span>
<span class="wc-name">The Viewer Plugin</span>
<button class="wc-check" data-ext="viewer" onclick="extCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">🧭</span><div><strong>Build this if</strong> you want to visualise report data in the browser - your own extension's columns, anyone else's, or any mix - with a chart the built-in widgets don't cover.</div></div>
</div>

<p>A viewer plugin is a small JavaScript module that renders a custom widget in the report viewer's sidebar, with full access to the report's data through an in-browser DuckDB instance (the table is always called <code>pp_data</code>). It exports one default object:</p>

```js
export default {
  id:    'my-widget',          // unique across all loaded plugins
  label: 'My Widget',          // shown in the sidebar widget list
  group: 'My Extension Name',  // optional - gives the widget its own sidebar section
  scope: 'image',              // optional - 'file' | 'image' | 'slice', shown as a badge
                               // describing what one datapoint in this widget represents

  requires(schema) {
    // return false to hide the widget when its columns are absent
    return schema.allCols.includes('my_column');
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT my_column, COUNT(*) AS cnt
      FROM pp_data
      ${ctx.where}
      GROUP BY 1 ORDER BY 2 DESC
    `);
    // write into `container` using plain DOM, Plotly (window.Plotly), or any CDN library
  },
};
```

<table>
<thead><tr><th><code>ctx</code> field</th><th>Type</th><th>Description</th></tr></thead>
<tbody>
<tr><td><code>ctx.queryRows(sql)</code></td><td><code>async → object[]</code></td><td>query returning plain JS objects</td></tr>
<tr><td><code>ctx.query(sql)</code></td><td><code>async → Arrow Table</code></td><td>raw Arrow result (for binary/blob columns)</td></tr>
<tr><td><code>ctx.querySample(cols, n)</code></td><td><code>async → object[]</code></td><td>sampled scalar shorthand</td></tr>
<tr><td><code>ctx.schema</code></td><td><code>object</code></td><td><code>{ metricCols, groupCols, dimensionInfo, allCols, blobCols }</code></td></tr>
<tr><td><code>ctx.state</code></td><td><code>object</code></td><td><code>{ palette, groupCol, filter, dimensions }</code></td></tr>
<tr><td><code>ctx.colorMap</code></td><td><code>object</code></td><td><code>{ groupValue: hexColor }</code> - matches the colors used everywhere else in the report</td></tr>
<tr><td><code>ctx.color.getColors(palette, n)</code></td><td><code>(string, number) -&gt; string[]</code></td><td><code>n</code> colors from the named palette - for ad-hoc groupings (e.g. a column other than the active group-by) not covered by <code>colorMap</code></td></tr>
<tr><td><code>ctx.color.getPaletteNames()</code></td><td><code>() -&gt; string[]</code></td><td>palette names accepted by <code>ctx.color.getColors</code></td></tr>
<tr><td><code>ctx.where</code></td><td><code>string</code></td><td>SQL <code>WHERE</code> clause for the active filter (or <code>''</code>) - merge with <code>AND</code> if your query needs its own</td></tr>
<tr><td><code>ctx.groups</code></td><td><code>string[]</code></td><td>distinct values of the active group column</td></tr>
<tr><td><code>ctx.filteredCount</code> / <code>ctx.totalRows</code></td><td><code>number</code></td><td>row counts</td></tr>
</tbody>
</table>

<p>Pixel HAI Watch ships <strong>two</strong> plugins on purpose - one per kind of data a loader can surface. <code>plugin_glow_by_depth.js</code> plots <code>glow_count</code> (computed by the processor, from real pixel data) against <code>depth_zone</code> (read straight from the loader's metadata), as a jittered scatter colored by site:</p>

```js
const DEPTH_ORDER = ['sunlit', 'twilight', 'midnight', 'abyss'];

export default {
  id:    'glow-by-depth',
  label: 'Glow Sightings by Depth',
  group: 'Pixel HAI Watch',
  scope: 'image',

  requires(schema) {
    return ['depth_zone', 'glow_count'].every(c => schema.allCols.includes(c));
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT "depth_zone" AS depth_zone, "imported_path_short" AS site, "glow_count" AS glows
      FROM pp_data
      WHERE "depth_zone" IS NOT NULL AND "glow_count" IS NOT NULL
        ${ctx.where ? 'AND ' + ctx.where.replace(/^WHERE\s+/i, '') : ''}
    `);

    const zones   = DEPTH_ORDER.filter(z => rows.some(r => r.depth_zone === z));
    const sites   = [...new Set(rows.map(r => r.site))].sort();
    const xJitter = () => (Math.random() - 0.5) * 0.5;   // keeps overlapping points visible

    // A scatter (rather than a box/violin) because each category holds only a
    // handful of points - exactly where distributional summaries would mislead.
    Plotly.newPlot(container, sites.map(site => {
      const sub = rows.filter(r => r.site === site);
      return {
        type: 'scatter', mode: 'markers', name: site,
        x: sub.map(r => zones.indexOf(r.depth_zone) + xJitter()),
        y: sub.map(r => Number(r.glows)),
        marker: { size: 12, color: ctx.colorMap[site] ?? '#888' },
      };
    }), { title: { text: 'How much bioluminescent glow shows up at each depth?' } });
  },
};
```

<p>Both plugins are listed in a small manifest, loaded automatically by the viewer:</p>

```json
{
  "name": "Pixel HAI Watch Extension",
  "plugins": ["./plugin_dives_logged.js", "./plugin_glow_by_depth.js"]
}
```

<div class="wc-flags">
<div class="wc-flag wc-flag-green"><span class="fi">✅</span><div>Both plugins declare <code>group: 'Pixel HAI Watch'</code>, so they get their own named section in the sidebar instead of being lumped under "Other Widgets" - a small touch that makes an extension feel like a first-class part of the report.</div></div>
<div class="wc-flag wc-flag-blue"><span class="fi">💡</span><div>See the <a href="https://github.com/ida-mdc/pixel-patrol/blob/main/viewer/README.md" target="_blank">viewer README</a> for the full plugin-writing guide, the complete <code>ctx</code> reference, and the extension-manifest format.</div></div>
</div>

</div>
</div>

---

<div class="wc" data-ext-req="" id="ext-run">
<div class="wc-head">
<span class="wc-icon">🚀</span>
<span class="wc-name">Run, Package &amp; Share</span>
<span class="wc-pill wc-pill-always">always relevant</span>
<button class="wc-check" data-ext="run" onclick="extCheck(this)" title="Mark as reviewed"></button>
</div>
<div class="wc-body">

<p>Pixel Patrol discovers loaders, processors, and viewer extensions through Python entry points - which only works if your package is installed in the <em>same environment</em> as <code>pixel_patrol_base</code>. So first, make sure you're in that environment, then install this package into it:</p>

```sh
uv pip install -e .
```

<p>Then try it locally - this (re)generates the tiny dataset (if missing), processes it with the custom loader and processor, and opens the viewer with both widgets loaded:</p>

```bash
uv run python create_and_show_report.py
```

<p>Once it works, two ways to get it in front of others:</p>

<div class="wc-shots two-col" style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
  <div class="wc-shot" style="border:1px solid var(--wc-border);border-radius:8px;padding:0.9rem 1rem">
    <div style="font-weight:700;margin-bottom:0.35rem">📦 Pip package</div>
    <div style="font-size:0.85rem;line-height:1.6">Because the JS viewer plugins are bundled <em>inside</em> the Python package, a recipient just installs it - <code>pip install pixel-patrol-hai-watch</code> - and any report opened with <code>serve_viewer(...)</code> picks up the plugins automatically. No extra arguments, no separate hosting.</div>
  </div>
  <div class="wc-shot" style="border:1px solid var(--wc-border);border-radius:8px;padding:0.9rem 1rem">
    <div style="font-weight:700;margin-bottom:0.35rem">🌐 GitHub Pages</div>
    <div style="font-size:0.85rem;line-height:1.6">No Python required on the recipient's side. A bundled GitHub Actions workflow deploys the <code>viewer/</code> folder; the manifest then lives at a public URL you can pass straight to the hosted viewer:<br><code>?extension=https://&lt;org&gt;.github.io/&lt;repo&gt;/extension.json</code> (repeat <code>&amp;extension=</code> to chain several).</div>
  </div>
</div>

<div class="wc-flags">
<div class="wc-flag wc-flag-blue"><span class="fi">🌱</span><div><strong>Ready to grow your own?</strong> Copy <code>examples/minimal-extension/</code>, decide which piece(s) you actually need (see the questions at the top of this page), update the <code>pyproject.toml</code> metadata, and replace the example identifiers with your own - one piece at a time. The protocols will tell you exactly what's still missing as you go, and nothing stops you from running an unfinished extension while you build it out.</div></div>
</div>

</div>
</div>

---

<script>
/* ── Setup toggle (mirrors the Exploring the Report tutorial) ── */
const extState = { loader: null, processor: null, viewer: null };

function extSetup(btn) {
  const key = btn.dataset.key;
  const val = btn.dataset.val;
  extState[key] = val;
  document.querySelectorAll('.wc-setup-btn[data-key="' + key + '"]').forEach(b => {
    b.classList.toggle('sel', b.dataset.val === val);
  });
  extApply();
}

function extApply() {
  document.querySelectorAll('.wc[data-ext-req]').forEach(card => {
    const req = card.dataset.extReq;
    card.classList.toggle('wc-unavailable', !!req && extState[req] === 'no');
  });
}

/* ── Progress tracker ─────────────────────────────────────── */
const EXT_TOTAL = 5;
const EXT_KEY   = 'pp-extension-progress';

function extCheck(btn) {
  const id = btn.dataset.ext;
  const on = btn.classList.toggle('checked');
  btn.textContent = on ? '✓' : '';
  const saved = JSON.parse(localStorage.getItem(EXT_KEY) || '{}');
  saved[id] = on;
  localStorage.setItem(EXT_KEY, JSON.stringify(saved));
  extUpdateProgress();
}

function extUpdateProgress() {
  const saved = JSON.parse(localStorage.getItem(EXT_KEY) || '{}');
  const done  = Object.values(saved).filter(Boolean).length;
  const pct   = Math.round(done / EXT_TOTAL * 100);
  const fill  = document.getElementById('ext-prog-fill');
  const label = document.getElementById('ext-prog-label');
  if (fill)  fill.style.width  = pct + '%';
  if (label) label.textContent = done + ' / ' + EXT_TOTAL + ' pieces reviewed';
}

function extInit() {
  const saved = JSON.parse(localStorage.getItem(EXT_KEY) || '{}');
  document.querySelectorAll('.wc-check').forEach(btn => {
    if (saved[btn.dataset.ext]) {
      btn.classList.add('checked');
      btn.textContent = '✓';
    }
  });
  extUpdateProgress();
}

document.addEventListener('DOMContentLoaded', extInit);
</script>
