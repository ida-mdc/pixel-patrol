# Processing

!!! tip "Used the launcher instead?"
    Double-click it, a browser tab opens, and you can set up your project and start processing from there - no terminal needed. The rest of this page is for the CLI workflow.

`pixel-patrol process` is the first step in the Pixel Patrol workflow. It scans your images and produces a `.parquet` report file containing everything Pixel Patrol knows about your dataset - file metadata, image dimensions, pixel statistics, quality metrics, and thumbnails.

Answer the questions below and we'll walk you through each decision together, building your command as we go. By the end you'll understand not just *what* to run, but *why* each flag is there.

!!! warning "Make sure you're inside a virtual environment"
    Before running any `pixel-patrol` command, activate the virtual environment where you installed it.
    If you followed the uv instructions, run:
    ```
    source .venv/bin/activate        # macOS / Linux
    .venv\Scripts\Activate.ps1       # Windows
    ```
    Not sure? Go through the [Installation tutorial](installation.md) first.

!!! tip "Explore all options first"
    Before going through the questions, it's worth running `pixel-patrol process --help` to see every available flag and its default. The wizard covers all of them, but `--help` gives you the full picture at a glance.

---

<div class="pp-wizard" id="pp-proc-wizard">

  <!-- Live command preview -->
  <div class="wiz-preview">
    <div class="wiz-preview-header">
      <span class="wiz-preview-label" id="proc-preview-label">Your command</span>
      <div class="wiz-mode-toggle">
        <button class="wiz-mode-btn wiz-mode-active" id="proc-mode-cli" onclick="procWiz.setMode('cli')">CLI</button>
        <button class="wiz-mode-btn" id="proc-mode-api" onclick="procWiz.setMode('api')">Python script</button>
      </div>
      <button class="wiz-copy-btn" id="proc-copy-btn" onclick="procWiz.copy()">Copy</button>
    </div>
    <pre class="wiz-code-pre" id="proc-cmd">pixel-patrol process &lt;path/to/images/&gt; \
  -o report.parquet</pre>
    <div class="wiz-callout wiz-hidden" id="proc-api-slurm-warning" style="margin:0 20px 16px">
      ⚠️ There's no Python-API equivalent for SLURM clusters - <code>pixel-patrol-slurm</code> is a CLI launcher
      around <code>dask_jobqueue.SLURMCluster</code>. Switch to the <strong>CLI</strong> tab above for a ready-to-run
      SLURM command, or see <a href="../processing.md#connecting-to-an-external-dask-cluster">Connecting to an external Dask cluster</a>
      to wire up a <code>SLURMCluster</code> by hand from a script.
    </div>
  </div>

  <!-- Q1: Base directory -->
  <div class="wiz-step wiz-visible" id="pws-base_dir">
    <div class="wiz-step-q">📁 Where are your images?</div>
    <div class="wiz-step-hint">
      This is the root folder of your dataset - the <code>BASE_DIRECTORY</code> argument in the command.
      Pixel Patrol will scan it recursively, so you don't need to list subdirectories separately.
      Use an absolute path (e.g. <code>/data/my-experiment/</code>) or a path relative to where you'll run the command.
      No images of your own yet? The repo ships a small example dataset at
      <a href="https://github.com/ida-mdc/pixel-patrol/tree/main/examples/datasets/WHOI_processed_color"><code>examples/datasets/WHOI_processed_color/</code></a> (40 plankton images, four tampered
      variants of the same originals, ~1.3 MB total) - point <code>BASE_DIRECTORY</code> there and
      follow along. It's the very dataset behind the <a href="https://ida-mdc.github.io/pixel-patrol/viewer/?data=../example.parquet" target="_blank">example report</a>
      used in the next tutorial.
    </div>
    <input class="wiz-input" type="text" id="pwi-base_dir"
           placeholder="/path/to/images/"
           oninput="procWiz.set('base_dir', this.value)">
  </div>

  <!-- Q2: Loader -->
  <div class="wiz-step wiz-hidden" id="pws-loader">
    <div class="wiz-step-q">🔬 What format are your images?</div>
    <div class="wiz-step-hint">
      Pixel Patrol uses <strong>loaders</strong> to open image files and extract their content. This sets the <code>--loader</code> flag
      and determines what ends up in your report: without a loader you only get basic file system info (names, sizes, extensions);
      with one you also get image dimensions, pixel type, acquisition metadata, and the pixel data needed for statistics and thumbnails.
      Choose the one that matches your file format.
    </div>
    <div class="wiz-options">
      <label class="wiz-option">
        <input type="radio" name="proc-loader" value="bioio"
               onchange="procWiz.pick('loader','bioio')">
        <div>
          <div class="wiz-opt-title">
            TIFF, CZI, ND2, LIF, PNG, JPG, or other common formats
            <code class="wiz-badge">--loader bioio</code>
          </div>
          <div class="wiz-opt-sub">Recommended for most microscopy and general image datasets</div>
        </div>
      </label>
      <label class="wiz-option">
        <input type="radio" name="proc-loader" value="zarr"
               onchange="procWiz.pick('loader','zarr')">
        <div>
          <div class="wiz-opt-title">
            Zarr datasets
            <code class="wiz-badge">--loader zarr</code>
          </div>
        </div>
      </label>
      <label class="wiz-option">
        <input type="radio" name="proc-loader" value="tifffile"
               onchange="procWiz.pick('loader','tifffile')">
        <div>
          <div class="wiz-opt-title">
            TIFF only - lightweight loader
            <code class="wiz-badge">--loader tifffile</code>
          </div>
          <div class="wiz-opt-sub">Faster when your dataset is exclusively TIFF files</div>
        </div>
      </label>
      <label class="wiz-option">
        <input type="radio" name="proc-loader" value=""
               onchange="procWiz.pick('loader','')">
        <div>
          <div class="wiz-opt-title">Just basic file info - no image reading</div>
          <div class="wiz-opt-sub">Collects file names, sizes, and extensions only. No loader required.</div>
        </div>
      </label>
    </div>
  </div>

  <!-- Q2b: File extensions (after loader chosen) -->
  <div class="wiz-step wiz-hidden" id="pws-file_exts">
    <div class="wiz-step-q">🗃️ Restrict to specific file extensions? <span class="wiz-optional">(optional)</span></div>
    <div class="wiz-step-hint">
      By default the loader processes all file formats it supports. If your folder contains mixed file types
      and you only want to process some of them, list the extensions here - each becomes a <code>-e</code> flag.
      Leave blank to process everything the loader supports.
      Example: <code>tif, nd2, czi</code>
    </div>
    <input class="wiz-input" type="text" id="pwi-file_exts"
           placeholder="tif, nd2, czi"
           oninput="procWiz.set('file_exts', this.value)">
  </div>

  <!-- Q3: Experimental conditions? -->
  <div class="wiz-step wiz-hidden" id="pws-conditions">
    <div class="wiz-step-q">🗂️ Do you have multiple experimental conditions or groups?</div>
    <div class="wiz-step-hint">
      Use this if your images are organized into subfolders - one per condition, batch, or timepoint.
      Specifying subfolders does two things: it <strong>limits processing to only those folders</strong> (others are ignored),
      and it sets each one as a <strong>labeled group</strong> in the report, shown in different colors for easy comparison.
      This grouping is the default - you can always regroup interactively in the viewer later.
      If you skip this, all images under the base folder are processed as one group.
    </div>
    <div class="wiz-options wiz-options-row">
      <label class="wiz-option">
        <input type="radio" name="proc-cond" value="no"
               onchange="procWiz.setConditions('no')">
        <div><div class="wiz-opt-title">No - process everything as one group</div></div>
      </label>
      <label class="wiz-option">
        <input type="radio" name="proc-cond" value="yes"
               onchange="procWiz.setConditions('yes')">
        <div><div class="wiz-opt-title">Yes - I have subfolders to compare</div></div>
      </label>
    </div>
  </div>

  <!-- Q3b: Subfolder names -->
  <div class="wiz-step wiz-hidden" id="pws-cond_names">
    <div class="wiz-step-q">Which subfolders should be compared?</div>
    <div class="wiz-step-hint">
      Comma-separated paths relative to your base directory - each becomes a <code>-p</code> flag and a labeled group.
      Can be immediate subfolders or deeper (e.g. <code>batch_1/control</code>). Only include the ones you want to compare.
      Example: <code>control, treated_a, treated_b</code>
    </div>
    <input class="wiz-input" type="text" id="pwi-cond_names"
           placeholder="control, treated_a, treated_b"
           oninput="procWiz.setConditionNames(this.value)">
  </div>

  <!-- Q4: Output path -->
  <div class="wiz-step wiz-hidden" id="pws-output">
    <div class="wiz-step-q">💾 Where should the output report be saved?</div>
    <div class="wiz-step-hint">
      A path (relative or absolute) for the output <code>.parquet</code> file - set by <code>-o</code>.
      This file holds all image metadata, pixel statistics, and thumbnails, and can be shared with collaborators
      who can open it in the <a href="https://ida-mdc.github.io/pixel-patrol/viewer/" target="_blank">online viewer</a> without installing anything.
    </div>
    <input class="wiz-input" type="text" id="pwi-output"
           value="report.parquet"
           oninput="procWiz.set('output', this.value || 'report.parquet')">
  </div>

  <!-- Q5: Project name -->
  <div class="wiz-step wiz-hidden" id="pws-name">
    <div class="wiz-step-q">🏷️ Give your project a name <span class="wiz-optional">(optional)</span></div>
    <div class="wiz-step-hint">
      Sets <code>--name</code>. Shown in the viewer header and embedded in the report file.
      If left empty, the name defaults to the base directory folder name.
    </div>
    <input class="wiz-input" type="text" id="pwi-name"
           placeholder="My Experiment"
           oninput="procWiz.set('name', this.value)">
  </div>

  <!-- Q5b: Description -->
  <div class="wiz-step wiz-hidden" id="pws-description">
    <div class="wiz-step-q">📝 Add a description <span class="wiz-optional">(optional)</span></div>
    <div class="wiz-step-hint">
      Sets <code>--description</code>. Free-form text shown below the title in the viewer and embedded in the report.
      Useful for recording what the dataset is, who processed it, or any caveats.
    </div>
    <input class="wiz-input" type="text" id="pwi-description"
           placeholder="Treated vs control, 3 replicates"
           oninput="procWiz.set('description', this.value)">
  </div>

  <!-- Q6: Cluster? -->
  <div class="wiz-step wiz-hidden" id="pws-cluster">
    <div class="wiz-step-q">🖥️ Are you running on an HPC cluster?</div>
    <div class="wiz-step-hint">
      Pixel Patrol processes images in parallel using <a href="https://www.dask.org/" target="_blank">Dask</a>.
      On a local machine it auto-detects a sensible number of workers based on your CPUs and RAM - no configuration needed.
      On a cluster you can harness many more resources, which makes a real difference for large datasets
      with thousands of images or very large volumes.
    </div>
    <div class="wiz-options wiz-options-row">
      <label class="wiz-option">
        <input type="radio" name="proc-cluster" value="no"
               onchange="procWiz.setCluster('no')">
        <div><div class="wiz-opt-title">No - running locally</div></div>
      </label>
      <label class="wiz-option">
        <input type="radio" name="proc-cluster" value="yes"
               onchange="procWiz.setCluster('yes')">
        <div><div class="wiz-opt-title">Yes - using a cluster</div></div>
      </label>
    </div>
  </div>

  <!-- Q6b: SLURM? -->
  <div class="wiz-step wiz-hidden" id="pws-slurm">
    <div class="wiz-step-q">Is your cluster managed by SLURM?</div>
    <div class="wiz-step-hint">
      SLURM is the most widely used job scheduler on HPC clusters.
      If your cluster uses it, <code>pixel-patrol-slurm</code> handles everything: it submits worker jobs,
      waits for them to come online, runs the processing, and cleans up - all in one command.
      If you're using a different setup (e.g. you already have a running Dask cluster), choose the second option
      and provide the scheduler address instead.
    </div>
    <div class="wiz-options wiz-options-row">
      <label class="wiz-option">
        <input type="radio" name="proc-slurm" value="yes"
               onchange="procWiz.setSlurm('yes')">
        <div><div class="wiz-opt-title">Yes - SLURM</div></div>
      </label>
      <label class="wiz-option">
        <input type="radio" name="proc-slurm" value="no"
               onchange="procWiz.setSlurm('no')">
        <div>
          <div class="wiz-opt-title">No - I have a Dask scheduler URL</div>
          <div class="wiz-opt-sub">e.g. from a manually started Dask cluster</div>
        </div>
      </label>
    </div>
  </div>

  <!-- Q6c: SLURM params -->
  <div class="wiz-step wiz-hidden" id="pws-slurm_params">
    <div class="wiz-step-q">SLURM cluster settings</div>
    <div class="wiz-callout">
      First, install the SLURM wrapper in the same environment:<br>
      <code>pip install pixel-patrol-slurm</code>
    </div>
    <div class="wiz-grid">
      <div class="wiz-field">
        <label>Number of jobs</label>
        <input class="wiz-input" type="number" id="pwi-slurm_jobs"
               value="4" min="1"
               oninput="procWiz.set('slurm_jobs', this.value || '4')">
      </div>
      <div class="wiz-field">
        <label>Cores per job</label>
        <input class="wiz-input" type="number" id="pwi-slurm_cores"
               value="4" min="1"
               oninput="procWiz.set('slurm_cores', this.value || '4')">
      </div>
      <div class="wiz-field">
        <label>Memory per job</label>
        <input class="wiz-input" type="text" id="pwi-slurm_mem"
               value="16GB"
               oninput="procWiz.set('slurm_mem', this.value || '16GB')">
      </div>
      <div class="wiz-field">
        <label>Partition <span class="wiz-optional">(optional)</span></label>
        <input class="wiz-input" type="text" id="pwi-slurm_partition"
               placeholder="gpu"
               oninput="procWiz.set('slurm_partition', this.value)">
      </div>
      <div class="wiz-field">
        <label>Wall time</label>
        <input class="wiz-input" type="text" id="pwi-slurm_walltime"
               value="02:00:00"
               oninput="procWiz.set('slurm_walltime', this.value || '02:00:00')">
      </div>
    </div>
  </div>

  <!-- Q6d: Dask scheduler URL -->
  <div class="wiz-step wiz-hidden" id="pws-scheduler_url">
    <div class="wiz-step-q">Dask scheduler URL</div>
    <div class="wiz-step-hint">The address of your running Dask scheduler, set by <code>--scheduler</code>.</div>
    <input class="wiz-input" type="text" id="pwi-scheduler_url"
           placeholder="tcp://hostname:8786"
           oninput="procWiz.set('scheduler_url', this.value)">
  </div>

  <!-- Done message -->
  <div class="wiz-done-msg" id="proc-done">
    <span id="proc-done-text">✓ Your command is ready above. Once it finishes, open the report with <code>pixel-patrol view report.parquet</code>.</span>
  </div>

  <!-- Advanced toggle -->
  <div id="pws-advanced-toggle" style="display:none;margin-top:1.5rem">
    <button class="wiz-adv-btn" id="adv-toggle-btn" onclick="procWiz.toggleAdvanced()">
      ▸ Show advanced options
    </button>
  </div>

  <!-- Advanced options section -->
  <div class="wiz-adv-section wiz-hidden" id="pws-advanced">

    <div class="wiz-adv-title">Advanced options</div>
    <div class="wiz-adv-intro">These options are rarely needed for a first run. They let you fine-tune processing behaviour for specific scenarios.</div>

    <!-- Slice size -->
    <div class="wiz-adv-field">
      <div class="wiz-adv-field-label">
        Slice size <code class="wiz-badge">--slice-size</code>
        <span class="wiz-optional"> (optional)</span>
      </div>
      <div class="wiz-adv-field-hint">
        Controls the per-dimension granularity of statistics in the report.
        By default, non-spatial dimensions (Z, T, C, S) each step by 1 - one row of stats per slice.
        Set a higher step (e.g. <code>Z=5</code>) for coarser, smaller output, or <code>-1</code> to collapse a
        dimension entirely. Only relevant if you have multidimensional data and care about per-slice statistics.
        Comma-separated for multiple dims. Example: <code>Z=5, C=1</code>
      </div>
      <input class="wiz-input" type="text" id="pwi-slice_size"
             placeholder="Z=5, C=1"
             oninput="procWiz.set('slice_size', this.value)">
    </div>

    <!-- Processors include -->
    <div class="wiz-adv-field">
      <div class="wiz-adv-field-label">
        Run only these processors <code class="wiz-badge">--processors-include</code>
        <span class="wiz-optional"> (optional)</span>
      </div>
      <div class="wiz-adv-field-hint">
        By default all installed processors run: <code>raster-basic</code>, <code>raster-histogram</code>,
        <code>thumbnail</code>, <code>raster-quality</code>, <code>raster-compression</code>.
        List specific ones here to run only those - takes precedence over exclude.
        Useful for speeding up processing when you only need a subset of metrics.
      </div>
      <input class="wiz-input" type="text" id="pwi-processors_include"
             placeholder="raster-basic, thumbnail"
             oninput="procWiz.set('processors_include', this.value)">
    </div>

    <!-- Processors exclude -->
    <div class="wiz-adv-field">
      <div class="wiz-adv-field-label">
        Skip these processors <code class="wiz-badge">--processors-exclude</code>
        <span class="wiz-optional"> (optional)</span>
      </div>
      <div class="wiz-adv-field-hint">
        Exclude specific processors while running all others. Ignored if <em>include</em> is set.
        Example: <code>raster-quality</code> to skip the quality metrics and speed up a large run.
      </div>
      <input class="wiz-input" type="text" id="pwi-processors_exclude"
             placeholder="raster-quality"
             oninput="procWiz.set('processors_exclude', this.value)">
    </div>

    <!-- Max workers (hidden for SLURM) -->
    <div class="wiz-adv-field" id="padv-max_workers">
      <div class="wiz-adv-field-label">
        Worker count <code class="wiz-badge">--max-workers</code>
        <span class="wiz-optional"> (optional)</span>
      </div>
      <div class="wiz-adv-field-hint">
        Number of parallel Dask workers (default: auto, based on CPUs and RAM).
        Lower this if processing causes out-of-memory errors. Use <code>1</code> to disable parallelism entirely,
        which is useful for debugging.
      </div>
      <input class="wiz-input" type="number" id="pwi-max_workers"
             placeholder="auto" min="1"
             oninput="procWiz.set('max_workers', this.value)">
    </div>

    <!-- MB per task -->
    <div class="wiz-adv-field">
      <div class="wiz-adv-field-label">
        MB per task <code class="wiz-badge">--mb-per-task</code>
        <span class="wiz-optional"> (default: 512)</span>
      </div>
      <div class="wiz-adv-field-hint">
        Memory budget per Dask task in MB. Controls how files are batched:
        <strong>increase</strong> (e.g. 2048) for datasets with many tiny files to reduce scheduling overhead;
        <strong>decrease</strong> (e.g. 128) for large 3D volumes or container files with large sub-images
        to keep individual tasks short and prevent memory spikes.
      </div>
      <input class="wiz-input" type="number" id="pwi-mb_per_task"
             placeholder="512" min="1"
             oninput="procWiz.set('mb_per_task', this.value)">
    </div>

    <!-- Max images per task -->
    <div class="wiz-adv-field">
      <div class="wiz-adv-field-label">
        Max images per task <code class="wiz-badge">--max-images-per-task</code>
        <span class="wiz-optional"> (default: 200)</span>
      </div>
      <div class="wiz-adv-field-hint">
        Maximum number of files (or sub-images) grouped into a single task. Lower this if individual tasks
        are timing out or you want finer-grained progress reporting.
      </div>
      <input class="wiz-input" type="number" id="pwi-max_images_per_task"
             placeholder="200" min="1"
             oninput="procWiz.set('max_images_per_task', this.value)">
    </div>

    <!-- Rows per part -->
    <div class="wiz-adv-field">
      <div class="wiz-adv-field-label">
        Rows per part <code class="wiz-badge">--rows-per-part</code>
        <span class="wiz-optional"> (default: 10000)</span>
      </div>
      <div class="wiz-adv-field-hint">
        Number of result rows buffered in memory before being flushed to a temporary file on disk.
        Only relevant for very large datasets where memory is tight. Rarely needs changing.
      </div>
      <input class="wiz-input" type="number" id="pwi-rows_per_part"
             placeholder="10000" min="1"
             oninput="procWiz.set('rows_per_part', this.value)">
    </div>

    <!-- Log file -->
    <div class="wiz-adv-field">
      <div class="wiz-adv-field-label">
        Write a debug log <code class="wiz-badge">--log-file</code>
      </div>
      <div class="wiz-adv-field-hint">
        Writes a detailed debug log file alongside the output parquet. Useful for diagnosing slow or failed runs.
      </div>
      <label class="wiz-option" style="max-width:260px">
        <input type="checkbox" id="pwi-log_file"
               onchange="procWiz.set('log_file', this.checked)">
        <div><div class="wiz-opt-title">Enable debug logging</div></div>
      </label>
    </div>

  </div>

</div>

<script>
const procWiz = {
  state: {
    base_dir:            '',
    loader:              null,
    file_exts:           '',
    cond_choice:         null,
    conditions:          [],
    output:              'report.parquet',
    name:                '',
    description:         '',
    cluster:             null,
    slurm:               null,
    slurm_jobs:          '4',
    slurm_cores:         '4',
    slurm_mem:           '16GB',
    slurm_partition:     '',
    slurm_walltime:      '02:00:00',
    scheduler_url:       '',
    adv_open:            false,
    slice_size:          '',
    processors_include:  '',
    processors_exclude:  '',
    max_workers:         '',
    mb_per_task:         '',
    max_images_per_task: '',
    rows_per_part:       '',
    log_file:            false,
    output_mode:         'cli',
  },

  show(id) {
    const el = document.getElementById('pws-' + id);
    if (el && el.classList.contains('wiz-hidden')) {
      el.classList.remove('wiz-hidden');
      el.classList.add('wiz-visible');
    }
  },

  hide(id) {
    const el = document.getElementById('pws-' + id);
    if (el) { el.classList.add('wiz-hidden'); el.classList.remove('wiz-visible'); }
  },

  highlightRadio(name, value) {
    document.querySelectorAll('input[name="' + name + '"]').forEach(r => {
      r.closest('.wiz-option').classList.toggle('wiz-selected', r.value === value);
    });
  },

  set(field, value) {
    this.state[field] = value;
    this.rebuild();
    this.reveal();
  },

  pick(field, value) {
    this.state[field] = value;
    this.highlightRadio('proc-' + field, value);
    this.rebuild();
    this.reveal();
  },

  reveal() {
    const s = this.state;

    if (s.base_dir && s.base_dir.trim()) this.show('loader');

    if (s.loader !== null) {
      this.show('file_exts');
      this.show('conditions');
    }

    if (s.cond_choice === 'yes') this.show('cond_names');
    if (s.cond_choice === 'no')  this.hide('cond_names');

    const condsDone = s.cond_choice === 'no' || s.cond_choice === 'yes';
    if (condsDone && s.loader !== null) {
      this.show('output');
      this.show('name');
      this.show('description');
      this.show('cluster');
    }

    if (s.cluster === 'yes') this.show('slurm');
    if (s.cluster === 'no')  {
      this.hide('slurm'); this.hide('slurm_params'); this.hide('scheduler_url');
    }
    if (s.cluster === 'yes' && s.slurm === 'yes') {
      this.show('slurm_params'); this.hide('scheduler_url');
    }
    if (s.cluster === 'yes' && s.slurm === 'no') {
      this.hide('slurm_params'); this.show('scheduler_url');
    }

    const isSlurm   = s.cluster === 'yes' && s.slurm === 'yes';
    const clusterDone = s.cluster === 'no' ||
                        (s.cluster === 'yes' && s.slurm !== null);
    if (condsDone && s.loader !== null && clusterDone) {
      const done = document.getElementById('proc-done');
      if (done) done.classList.add('wiz-visible');
      const tog = document.getElementById('pws-advanced-toggle');
      if (tog) tog.style.display = 'block';
    }

    // Hide max-workers field when using SLURM (irrelevant)
    const mw = document.getElementById('padv-max_workers');
    if (mw) mw.style.display = isSlurm ? 'none' : '';
  },

  setConditions(val) {
    this.state.cond_choice = val;
    this.highlightRadio('proc-cond', val);
    this.rebuild();
    this.reveal();
  },

  setConditionNames(val) {
    this.state.conditions = val.split(',').map(s => s.trim()).filter(Boolean);
    this.rebuild();
  },

  setCluster(val) {
    this.state.cluster = val;
    this.highlightRadio('proc-cluster', val);
    this.rebuild();
    this.reveal();
  },

  setSlurm(val) {
    this.state.slurm = val;
    this.highlightRadio('proc-slurm', val);
    this.rebuild();
    this.reveal();
  },

  setMode(mode) {
    this.state.output_mode = mode;
    document.getElementById('proc-mode-cli').classList.toggle('wiz-mode-active', mode === 'cli');
    document.getElementById('proc-mode-api').classList.toggle('wiz-mode-active', mode === 'api');
    const label = document.getElementById('proc-preview-label');
    if (label) label.textContent = (mode === 'api') ? 'Your script' : 'Your command';
    this.rebuild();
  },

  toggleAdvanced() {
    this.state.adv_open = !this.state.adv_open;
    const btn = document.getElementById('adv-toggle-btn');
    const sec = document.getElementById('pws-advanced');
    if (this.state.adv_open) {
      sec.classList.remove('wiz-hidden'); sec.classList.add('wiz-visible');
      btn.textContent = '▾ Hide advanced options';
    } else {
      sec.classList.add('wiz-hidden'); sec.classList.remove('wiz-visible');
      btn.textContent = '▸ Show advanced options';
    }
  },

  rebuild() {
    const s = this.state;
    const base    = s.base_dir && s.base_dir.trim() ? s.base_dir.trim() : '<path/to/images/>';
    const out     = s.output || 'report.parquet';
    const isSlurm = s.cluster === 'yes' && s.slurm === 'yes';
    const hasDask = s.cluster === 'yes' && s.slurm === 'no';

    const text = (s.output_mode === 'api')
      ? this.buildApiScript(s, base, out, isSlurm, hasDask)
      : this.buildCliCommand(s, base, out, isSlurm, hasDask);
    document.getElementById('proc-cmd').textContent = text;

    const warn = document.getElementById('proc-api-slurm-warning');
    if (warn) warn.classList.toggle('wiz-hidden', !(s.output_mode === 'api' && isSlurm));

    const doneText = document.getElementById('proc-done-text');
    if (doneText) {
      doneText.innerHTML = (s.output_mode === 'api')
        ? '✓ Your script is ready above - save it (e.g. <code>run_processing.py</code>) and run it with <code>python run_processing.py</code>. It already opens the viewer at the end via <code>api.view(project)</code>.'
        : '✓ Your command is ready above. Once it finishes, open the report with <code>pixel-patrol view report.parquet</code>.';
    }
  },

  buildCliCommand(s, base, out, isSlurm, hasDask) {
    // Build pixel-patrol process args (shared between local and SLURM)
    const pp = [];
    pp.push('-o ' + out);
    if (s.loader) pp.push('--loader ' + s.loader);
    if (s.file_exts && s.file_exts.trim())
      s.file_exts.split(',').map(e => e.trim()).filter(Boolean)
        .forEach(e => pp.push('-e ' + e));
    if (s.conditions && s.conditions.length)
      s.conditions.forEach(c => pp.push('-p ' + c));
    if (s.name && s.name.trim())
      pp.push('--name "' + s.name.trim() + '"');
    if (s.description && s.description.trim())
      pp.push('--description "' + s.description.trim() + '"');
    if (hasDask && s.scheduler_url && s.scheduler_url.trim())
      pp.push('--scheduler ' + s.scheduler_url.trim());

    // Advanced
    if (s.slice_size && s.slice_size.trim())
      s.slice_size.split(',').map(e => e.trim()).filter(Boolean)
        .forEach(sz => pp.push('--slice-size ' + sz));
    if (s.processors_include && s.processors_include.trim())
      s.processors_include.split(',').map(e => e.trim()).filter(Boolean)
        .forEach(p => pp.push('--processors-include ' + p));
    if (s.processors_exclude && s.processors_exclude.trim())
      s.processors_exclude.split(',').map(e => e.trim()).filter(Boolean)
        .forEach(p => pp.push('--processors-exclude ' + p));
    if (!isSlurm && s.max_workers) pp.push('--max-workers ' + s.max_workers);
    if (s.mb_per_task)         pp.push('--mb-per-task ' + s.mb_per_task);
    if (s.max_images_per_task) pp.push('--max-images-per-task ' + s.max_images_per_task);
    if (s.rows_per_part)       pp.push('--rows-per-part ' + s.rows_per_part);
    if (s.log_file)            pp.push('--log-file');

    let lines = [];
    if (isSlurm) {
      lines.push('pixel-patrol-slurm');
      lines.push('  --jobs '     + (s.slurm_jobs    || '4'));
      lines.push('  --cores '    + (s.slurm_cores   || '4'));
      lines.push('  --memory '   + (s.slurm_mem     || '16GB'));
      if (s.slurm_partition && s.slurm_partition.trim())
        lines.push('  --partition ' + s.slurm_partition.trim());
      lines.push('  --walltime ' + (s.slurm_walltime || '02:00:00'));
      lines.push('  --');
      lines.push('  ' + base);
      pp.forEach(a => lines.push('  ' + a));
    } else {
      lines.push('pixel-patrol process ' + base);
      pp.forEach(a => lines.push('  ' + a));
    }

    return lines.join(' \\\n');
  },

  buildApiScript(s, base, out, isSlurm, hasDask) {
    const pyBase = (base === '<path/to/images/>') ? '/path/to/images/' : base;
    const projName = (s.name && s.name.trim())
      ? s.name.trim()
      : (pyBase.replace(/\/+$/, '').split('/').filter(Boolean).pop() || 'My Project');

    const lines = [];
    lines.push('from pixel_patrol_base import api');
    if (hasDask) lines.push('from dask.distributed import Client');
    lines.push('');
    lines.push('project = api.create_project(');
    lines.push('    "' + projName + '",');
    lines.push('    base_dir="' + pyBase + '",');
    if (s.loader) lines.push('    loader="' + s.loader + '",');
    lines.push('    output_path="' + out + '",');
    lines.push(')');

    if (s.conditions && s.conditions.length) {
      const condList = s.conditions.map(c => '"' + c + '"').join(', ');
      lines.push('api.add_paths(project, [' + condList + '])');
    }

    // Build api.process_files kwargs
    const kwargs = [];
    if (s.file_exts && s.file_exts.trim()) {
      const exts = s.file_exts.split(',').map(e => e.trim()).filter(Boolean)
        .map(e => '"' + e + '"').join(', ');
      if (exts) kwargs.push('selected_file_extensions={' + exts + '}');
    }
    if (s.processors_include && s.processors_include.trim()) {
      const incl = s.processors_include.split(',').map(e => e.trim()).filter(Boolean)
        .map(e => '"' + e + '"').join(', ');
      if (incl) kwargs.push('processors_included={' + incl + '}');
    }
    if (s.processors_exclude && s.processors_exclude.trim()) {
      const excl = s.processors_exclude.split(',').map(e => e.trim()).filter(Boolean)
        .map(e => '"' + e + '"').join(', ');
      if (excl) kwargs.push('processors_excluded={' + excl + '}');
    }
    if (s.slice_size && s.slice_size.trim()) {
      const pairs = s.slice_size.split(',').map(e => e.trim()).filter(Boolean)
        .map(pair => {
          const [dim, val] = pair.split('=').map(p => p.trim());
          return '"' + dim + '": ' + val;
        }).join(', ');
      if (pairs) kwargs.push('slice_size={' + pairs + '}');
    }
    if (!isSlurm && s.max_workers) kwargs.push('max_workers=' + s.max_workers);
    if (s.mb_per_task)         kwargs.push('mb_per_task=' + s.mb_per_task);
    if (s.max_images_per_task) kwargs.push('max_images_per_task=' + s.max_images_per_task);
    if (s.rows_per_part)       kwargs.push('rows_per_part=' + s.rows_per_part);
    if (s.description && s.description.trim())
      kwargs.push('description="' + s.description.trim() + '"');
    if (s.log_file) kwargs.push('log_file=True');

    lines.push('');

    if (isSlurm) {
      lines.push('# No Python-API equivalent for SLURM clusters - see the note below.');
    } else {
      const callLines = [];
      if (kwargs.length) {
        callLines.push('api.process_files(');
        callLines.push('    project,');
        kwargs.forEach(k => callLines.push('    ' + k + ','));
        callLines.push(')');
      } else {
        callLines.push('api.process_files(project)');
      }

      if (hasDask) {
        const url = (s.scheduler_url && s.scheduler_url.trim()) || 'tcp://hostname:8786';
        lines.push('with Client("' + url + '"):');
        callLines.forEach(l => lines.push('    ' + l));
      } else {
        callLines.forEach(l => lines.push(l));
      }

      lines.push('');
      lines.push('api.view(project)');
    }

    return lines.join('\n');
  },

  copy() {
    const text = document.getElementById('proc-cmd').textContent;
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('proc-copy-btn');
      const prev = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = prev; }, 2000);
    });
  }
};

document.addEventListener('DOMContentLoaded', function() {
  const outEl = document.getElementById('pwi-output');
  if (outEl) procWiz.state.output = outEl.value;
});
</script>

---

!!! tip "Next step"
    Once processing finishes, open your report with:
    ```bash
    pixel-patrol view report.parquet
    ```
    Or drag the `.parquet` file into the [online viewer](https://ida-mdc.github.io/pixel-patrol/viewer/) - no install needed on the recipient's side.
