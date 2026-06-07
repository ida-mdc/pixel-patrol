# Sharing a Report

A Pixel Patrol report is a single `.parquet` file - everything needed to reconstruct the interactive viewer is in that one file. This page covers all the ways to share and distribute it.

---

## At a glance

| Method | Recipient needs | Best for |
|---|---|---|
| [Drag & drop into hosted viewer](#drag-and-drop-into-the-hosted-viewer) | Nothing | Quick shares, small/medium files |
| [`pixel-patrol view`](#share-the-parquet-run-view-locally) | Pixel Patrol installed | Large files (5 GB+), self-sufficient recipients |
| [Self-contained HTML file](#build-a-self-contained-html-viewer) | A browser | Sending a viewer that works offline |
| [Static hosting + URL](#host-on-a-static-server) | Nothing (public link) | Publishing results, team dashboards |
| [Filtered parquet](#share-a-filtered-subset) | Either of the above | Sharing a curated data subset |

---

## Drag and drop into the hosted viewer

No installation required on either end. Send your collaborator the `.parquet` file and point them to:

**[ida-mdc.github.io/pixel-patrol/viewer/](https://ida-mdc.github.io/pixel-patrol/viewer/)**

They drag the file into the browser and the full interactive viewer loads. Everything runs locally in their browser - no data is uploaded anywhere.

!!! warning "File size limit for browser-based viewing"
    The hosted viewer runs DuckDB in a browser WebAssembly context, which is limited by available browser memory. Files under 1-2 GB typically work fine; very large files (5 GB+) may fail to load or become slow. For large reports, use `pixel-patrol view` instead.

---

## Share the parquet - run view locally

If your collaborator has Pixel Patrol installed, they can open any `.parquet` file directly:

```bash
pixel-patrol view report.parquet
```

This starts a local server with native DuckDB (no browser memory limit), making it fast even for large reports. This is the recommended method for files above a few gigabytes.

---

## Build a self-contained HTML viewer

You can package the viewer into a single HTML file that works with any parquet:

```bash
pixel-patrol build-viewer-html -o viewer.html
```

Send both `viewer.html` and `report.parquet` to your collaborator. They open `viewer.html` in any browser, use the file picker to load the parquet, and get the full interactive experience with no installation.

!!! note "The HTML file contains the viewer, not the data"
    `viewer.html` is the application shell - your collaborator still needs the `.parquet` file alongside it. The two files can be in the same folder; the viewer will offer a file picker on load.

---

## Host on a static server

If you want to publish results online - on GitHub Pages, an institutional server, an S3 bucket, or any static host - you can deploy the viewer as a site folder and link directly to your parquet:

**Step 1 - Build a viewer site folder:**
```bash
pixel-patrol build-viewer-html -o my-report-site/
```

This creates a `my-report-site/` directory with `index.html` and all viewer assets.

**Step 2 - Upload your parquet alongside the site** (or to any public URL).

**Step 3 - Link to the viewer with a `?data=` parameter:**
```
https://your-host.com/my-report-site/?data=https://your-host.com/report.parquet
```

The viewer fetches the parquet from the URL, so the parquet can be hosted anywhere that is publicly reachable (S3, GitHub releases, institutional data storage).

**GitHub Pages example:**

```bash
# Build viewer into your gh-pages output
pixel-patrol build-viewer-html -o docs/viewer/

# Add parquet to the same repo (or reference an external URL)
cp report.parquet docs/viewer/

# After deploying to GitHub Pages, share:
# https://your-org.github.io/your-repo/viewer/?data=https://your-org.github.io/your-repo/viewer/report.parquet
```

!!! tip "The hosted Pixel Patrol viewer also accepts a `?data=` URL"
    If your parquet is publicly accessible, you don't even need to deploy your own viewer:

    ```
    https://ida-mdc.github.io/pixel-patrol/viewer/?data=https://your-server.com/report.parquet
    ```

    Anyone with the link can open the full interactive report immediately.

---

## Share a filtered subset

After exploring your report, you often want to share a curated version - for example, after filtering out low-quality images or focusing on one condition. The viewer makes this easy.

1. **Apply your filters** using the sidebar (by quality metric, condition, file type, etc.).
2. **Save as Parquet** - this exports the current filtered view as a new `.parquet` file.
3. Share that parquet via any of the methods above.

The recipient gets a report containing only the images that passed your filters, with all the same widgets and interactivity. This is useful for:

- Sharing a "clean" dataset with a collaborator after removing outliers
- Sending a team member only the subset relevant to their analysis
- Creating a publishable report that excludes low-quality acquisitions

You can also **Save as CSV** to export the filtered data as a spreadsheet - the same table as the parquet, in a plain, human-readable format. Use it to build an include or exclude list for downstream processing (e.g. keeping only images whose `laplacian_variance` is above a threshold, or dropping the ones below it), or just open it yourself to browse and sort the full table in a human-readable format.

---

## Processing for sharing

If you're collecting data specifically to share a report (rather than running on your local machine), a couple of flags are worth knowing:

```bash
# Set a meaningful project name and description for the report header
pixel-patrol process /data/my_experiment \
    -o experiment_report.parquet \
    --name "Experiment 42 - Drug Screen" \
    --description "Round 1, n=384, 20x brightfield. Processed 2026-06-07."
```

The name and description appear at the top of every viewer session. Recipients don't have to guess what they're looking at.
