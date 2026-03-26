"""
GUI regression tests for the launcher app processing flow.

These tests require chromedriver and are skipped when it is not available.
They exercise the full browser interaction: clicking buttons, watching the
progress banner appear, verifying the report is added to the list after
processing, and confirming the sort order (newest first).
"""
import json
import shutil
import time
from pathlib import Path

import pytest

from pixel_patrol_base import api

_REQUIRES_CHROME = pytest.mark.skipif(
    shutil.which("chromedriver") is None,
    reason="chromedriver is not available in PATH",
)


@_REQUIRES_CHROME
def test_launcher_open_report_shows_report_page_with_data(
    dash_duo, project_with_all_data, tmp_path, monkeypatch
):
    """
    GUI regression test for the launcher-based report viewer.

    Steps:
    - Export a project to a ZIP.
    - Create a reports_index.json pointing at that ZIP in a temp reports dir.
    - Start create_processing_app() with PIXEL_PATROL_REPORTS_DIR set to that dir.
    - Click the first \"Open\" link.
    - Assert that we land on the report page and that it renders the report
      layout (header + at least one widget container).
    """
    export_path = tmp_path / "launcher_e2e.zip"
    api.export_project(project_with_all_data, export_path)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    index_file = reports_dir / "reports_index.json"
    index_file.write_text(
        json.dumps(
            [
                {
                    "zip_path": str(export_path.resolve()),
                    "base_dir": str(export_path.parent.resolve()),
                    "subpaths": [],
                    "loader": "",
                    "file_extensions": "",
                    "created_at": "2025-01-01T00:00:00",
                }
            ]
        )
    )

    monkeypatch.setenv("PIXEL_PATROL_REPORTS_DIR", str(reports_dir))
    # Import the launcher module and explicitly patch its global paths so this
    # test is robust even if other tests have already imported it with
    # different defaults.
    import pixel_patrol_base.processing_dashboard as lapp
    monkeypatch.setattr(lapp, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(lapp, "INDEX_FILE", index_file)

    app = lapp.create_processing_app()
    dash_duo.start_server(app)

    # Wait for the Reports heading to render on the launcher home page
    dash_duo.wait_for_text_to_equal("h4", "Pixel Patrol Reports", timeout=10)

    # Find the first "Open" link and click it.
    # The launcher uses html.A with href="/report?zip=...".
    # On slower CI runners (especially Windows) the reports list callback can
    # lag slightly behind the header render, so we poll for a short period.
    def _find_open_links():
        links = dash_duo.find_elements("a")
        return [a for a in links if "report?zip=" in (a.get_attribute("href") or "")]

    deadline = time.time() + 10
    open_links = []
    while time.time() < deadline and not open_links:
        open_links = _find_open_links()
        if open_links:
            break
        time.sleep(0.5)

    assert open_links, "Expected at least one Open link pointing to /report?zip=..."

    href = open_links[0].get_attribute("href")
    assert href, "Open link has no href"

    # Navigate directly to the report URL rather than clicking (which may be
    # intercepted by transient UI such as the Add Report modal).
    dash_duo.driver.get(href)

    # After navigation, we should be on the report page. Wait for the main
    # Pixel Patrol header to appear so we know the layout mounted.
    dash_duo.wait_for_text_to_equal("h1", "Pixel Patrol", timeout=20)

    # Then wait explicitly for at least one Plotly graph to appear. If this
    # times out, it indicates callbacks are not successfully populating data
    # (e.g. due to a loop or silent error), which should fail the test.
    dash_duo.wait_for_element(".js-plotly-plot", timeout=30)
    graphs = dash_duo.find_elements(".js-plotly-plot")
    assert graphs, "Expected at least one Plotly graph (.js-plotly-plot) on the report page"


@_REQUIRES_CHROME
def test_processing_progress_banner_appears_and_report_added_to_list(
    dash_duo, temp_test_dirs, tmp_path, monkeypatch
):
    """
    Regression test for three processing-flow bugs:

    1. Progress banner must appear immediately when "Run Processing" is clicked,
       not only after processing completes.
    2. The new report card must appear in the list on the home page after
       processing finishes, without requiring a restart.
    3. The newest report must appear at the top of the list.

    The test uses a real (small) dataset from temp_test_dirs so that the full
    api.process_files → api.export_project flow runs and produces a valid ZIP.
    """
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    # Pre-populate index with one old report so we can verify sort order.
    old_zip = tmp_path / "old_report.zip"
    old_zip.write_bytes(b"fake")  # content doesn't matter for list rendering
    old_entry = {
        "zip_path": str(old_zip.resolve()),
        "base_dir": str(tmp_path),
        "subpaths": [],
        "loader": "",
        "file_extensions": "",
        "created_at": "2020-01-01T00:00:00",
    }
    (reports_dir / "reports_index.json").write_text(json.dumps([old_entry]))

    monkeypatch.setenv("PIXEL_PATROL_REPORTS_DIR", str(reports_dir))
    # Point the output ZIPs into the tmp reports dir so _add_report_to_index
    # writes a ZIP that actually exists (we re-import constants after monkeypatch).
    import pixel_patrol_base.processing_dashboard as lapp
    monkeypatch.setattr(lapp, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(lapp, "INDEX_FILE", reports_dir / "reports_index.json")

    app = lapp.create_processing_app()
    dash_duo.start_server(app)

    # ── Home page loads ──────────────────────────────────────────────────────
    dash_duo.wait_for_text_to_equal("h4", "Pixel Patrol Reports", timeout=10)

    # ── Open the "Add Report" modal ──────────────────────────────────────────
    add_btn = dash_duo.find_element("#add-report-btn")
    add_btn.click()
    dash_duo.wait_for_element("#add-report-modal .modal-body", timeout=5)

    # ── Fill in the base directory (one of temp_test_dirs) ──────────────────
    base_input = dash_duo.find_element("#add-base-dir")
    base_input.send_keys(str(temp_test_dirs[0]))

    # ── Click "Run Processing" ───────────────────────────────────────────────
    run_btn = dash_duo.find_element("#add-run-btn")
    run_btn.click()

    # Bug 1: progress banner must appear before processing completes.
    # We check for the info alert that contains "Processing in progress".
    # The banner is rendered directly by toggle_modal_and_start_run so it
    # should be visible almost immediately.
    dash_duo.wait_for_element(".alert-info", timeout=5)
    banner = dash_duo.find_element(".alert-info")
    assert "Processing" in (banner.text or ""), (
        "Progress banner with 'Processing' text must appear immediately after "
        "clicking Run, but it was not found. Banner text: " + repr(banner.text)
    )

    # ── Wait for processing to complete (success banner appears) ─────────────
    dash_duo.wait_for_element(".alert-success", timeout=120)

    # Bug 2: new report must appear in the list without restart.
    # The reports list is updated by on_processing_complete via reports-index-store.
    # We look for a card that has base-dir text from our input.
    def _find_new_report_card():
        cards = dash_duo.find_elements(".card")
        for card in cards:
            if temp_test_dirs[0].name in (card.text or ""):
                return card
        return None

    # Poll up to 10s for the new card to appear.
    deadline = time.time() + 10
    new_card = None
    while time.time() < deadline:
        new_card = _find_new_report_card()
        if new_card:
            break
        time.sleep(0.5)

    assert new_card is not None, (
        "New report card should appear in the list after processing completes "
        "without requiring a restart. Card was not found."
    )

    # Bug 3: newest report must be at the top (before the old 2020 entry).
    # Each report card shows its date as plain text; use the reports-list container
    # to scope to report cards only (excludes modal and other cards).
    report_cards = dash_duo.find_elements("#reports-list .card")
    card_texts = [c.text for c in report_cards]
    assert len(card_texts) >= 2, f"Expected at least 2 report cards, got: {card_texts}"
    # The new report has created_at > 2020, so it must appear before the old entry.
    old_idx = next((i for i, t in enumerate(card_texts) if "2020" in t), None)
    new_idx = next((i for i, t in enumerate(card_texts) if "2020" not in t and t.strip()), None)
    assert new_idx is not None and old_idx is not None, (
        f"Could not identify new vs old report by card text. Texts: {card_texts}"
    )
    assert new_idx < old_idx, (
        f"Newest report (index {new_idx}) must appear before oldest (index {old_idx}). "
        f"Card texts in DOM order: {card_texts}"
    )


@_REQUIRES_CHROME
def test_progress_banner_not_shown_on_report_page(
    dash_duo, project_with_all_data, tmp_path, monkeypatch
):
    """
    The processing-status banner must NOT be visible when viewing a report.

    Regression: after moving the banner to the static layout (so it is always
    in the DOM), it was incorrectly shown above the report page.  The fix adds
    a URL check in show_processing_banner that hides it when pathname=/report.
    """
    export_path = tmp_path / "report_banner_test.zip"
    api.export_project(project_with_all_data, export_path)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "reports_index.json").write_text(json.dumps([{
        "zip_path": str(export_path.resolve()),
        "base_dir": str(tmp_path),
        "subpaths": [],
        "loader": "",
        "file_extensions": "",
        "created_at": "2025-06-01T12:00:00",
    }]))

    monkeypatch.setenv("PIXEL_PATROL_REPORTS_DIR", str(reports_dir))
    import pixel_patrol_base.processing_dashboard as lapp
    monkeypatch.setattr(lapp, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(lapp, "INDEX_FILE", reports_dir / "reports_index.json")

    # Seed a completed processing state so the banner *would* show on home page.
    monkeypatch.setattr(lapp, "_processing_state", {
        "status": "completed",
        "progress": 100,
        "message": "Done",
        "error": None,
        "output_zip": str(export_path.resolve()),
    })

    app = lapp.create_processing_app()
    dash_duo.start_server(app)

    # Navigate directly to the report page.
    from urllib.parse import quote
    report_url = dash_duo.server_url + f"/report?zip={quote(str(export_path.resolve()))}"
    dash_duo.driver.get(report_url)

    # Wait for the report layout to mount.
    dash_duo.wait_for_element(".js-plotly-plot", timeout=30)

    # The success banner must NOT appear on the report page.
    banners = dash_duo.find_elements(".alert-success")
    assert not banners, (
        "Processing-status banner must be hidden on the /report page, "
        "but an .alert-success element was found: "
        + repr([b.text for b in banners])
    )

