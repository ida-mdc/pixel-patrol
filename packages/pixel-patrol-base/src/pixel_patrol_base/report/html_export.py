"""
HTML export functionality for Pixel Patrol reports.

This module provides functionality to export the report dashboard as a static HTML file
using Playwright automation.
"""
import logging
import threading
import time
import socket
from pathlib import Path
from typing import Optional
from threading import Event

logger = logging.getLogger(__name__)


def export_html_from_dashboard(
    app,
    output_path: Path,
    host: str = "127.0.0.1",
    port: int = None,
    timeout: int = 120,
) -> None:
    """
    Export the report dashboard as a static HTML file using Playwright automation.
    
    This function:
    1. Launches the dashboard app in a background thread
    2. Uses Playwright to navigate to the dashboard in headless mode
    3. Waits for the page to fully render
    4. Extracts the HTML and applies transformations (reusing the JavaScript code)
    5. Saves the processed HTML to the specified path
    6. Shuts down the server
    
    Args:
        app: The Dash app instance to export
        output_path: Path where the HTML file should be saved
        host: Host address for the server (default: 127.0.0.1)
        port: Port number for the server (default: None, auto-assigned)
        timeout: Maximum time to wait for the export (seconds, default: 120)
    
    Raises:
        ImportError: If Playwright is not installed
        TimeoutError: If the export takes too long
        RuntimeError: If the export fails
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    except ImportError:
        raise ImportError(
            "Playwright is required for HTML export. Install it with: "
            "pip install playwright && playwright install chromium"
        )
    
    # Find a free port if not specified
    if port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
    
    url = f"http://{host}:{port}"
    
    # Start server in background thread
    def run_server():
        try:
            app.run(debug=False, host=host, port=port, use_reloader=False)
        except Exception as e:
            logger.error(f"Server error: {e}")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready by checking if port is listening
    logger.info("Waiting for server to start...")
    max_wait = 30
    wait_interval = 0.5
    waited = 0
    while waited < max_wait:
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1)
            result = test_socket.connect_ex((host, port))
            test_socket.close()
            if result == 0:
                logger.info(f"Server is ready on {url}")
                break
        except Exception:
            pass
        time.sleep(wait_interval)
        waited += wait_interval
    else:
        raise RuntimeError(f"Server failed to start on {host}:{port} within {max_wait} seconds")
    
    # Give server a moment to fully initialize
    time.sleep(2)
    
    try:
        with sync_playwright() as p:
            # Launch browser with a viewport size matching the dashboard's max width (1200px)
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1200, "height": 1080})
            page = context.new_page()
            
            logger.info(f"Navigating to {url}")
            # Use networkidle for initial load to ensure all resources are fetched
            page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            
            # Wait for the page to fully load
            logger.info("Waiting for page to load...")
            try:
                page.wait_for_selector("body", timeout=30000)
            except PlaywrightTimeoutError:
                raise RuntimeError("Page failed to load within timeout")
            
            # Wait for network to be idle again (in case Dash loaded additional resources)
            logger.info("Waiting for network to be idle...")
            try:
                page.wait_for_load_state("networkidle", timeout=60000)
            except PlaywrightTimeoutError:
                logger.warning("Network did not become idle, continuing anyway...")
            
            # Wait for DOM to be stable (no rapid changes)
            logger.info("Waiting for DOM to stabilize...")
            time.sleep(1)
            
            # Wait for Dash callbacks to complete by checking for specific elements
            logger.info("Waiting for Dash callbacks to complete...")
            try:
                # Wait for at least one widget/content element to appear
                page.wait_for_selector(".dash-graph, .card, .row", timeout=30000)
                
                # Wait for Dash to finish initial rendering by checking if _dash-renderer is ready
                page.wait_for_function(
                    "() => window.dash_clientside !== undefined || document.querySelector('.dash-graph') !== null",
                    timeout=30000
                )
            except PlaywrightTimeoutError:
                logger.warning("Dash content may not be fully loaded, continuing anyway...")
            
            # Wait for Plotly charts to render
            logger.info("Waiting for Plotly charts to render...")
            max_plot_wait = 60  # Maximum seconds to wait for plots (increased for large reports)
            plot_wait_interval = 0.5
            plot_waited = 0
            consecutive_ready_checks = 0
            required_consecutive = 3  # Require 3 consecutive checks to ensure stability
            
            while plot_waited < max_plot_wait:
                # Check if plots are rendered by looking for SVG elements
                plot_status = page.evaluate("""
                    () => {
                        const plotDivs = document.querySelectorAll('.js-plotly-plot');
                        if (plotDivs.length === 0) {
                            return { total: 0, rendered: 0, ready: false };
                        }
                        
                        // Check if plots have rendered SVGs with actual content
                        let renderedCount = 0;
                        plotDivs.forEach(div => {
                            const svg = div.querySelector('svg.main-svg');
                            if (svg) {
                                // Check for actual plot content (not just empty SVG)
                                const hasContent = svg.querySelector('g') || svg.querySelector('path') || svg.querySelector('rect');
                                if (hasContent) {
                                    renderedCount++;
                                }
                            }
                        });
                        
                        // Consider ready if all plots are rendered, or if we have at least 80% for large reports
                        const threshold = plotDivs.length <= 5 ? plotDivs.length : Math.max(1, Math.ceil(plotDivs.length * 0.8));
                        const ready = renderedCount >= threshold && renderedCount > 0;
                        
                        return { 
                            total: plotDivs.length, 
                            rendered: renderedCount, 
                            ready: ready 
                        };
                    }
                """)
                
                if plot_status['ready']:
                    consecutive_ready_checks += 1
                    if consecutive_ready_checks >= required_consecutive:
                        logger.info(f"Plotly charts are rendered ({plot_status['rendered']}/{plot_status['total']} plots)")
                        break
                else:
                    consecutive_ready_checks = 0
                    if plot_status['total'] > 0:
                        logger.debug(f"Waiting for plots: {plot_status['rendered']}/{plot_status['total']} rendered")
                
                time.sleep(plot_wait_interval)
                plot_waited += plot_wait_interval
            
            if plot_waited >= max_plot_wait:
                logger.warning(f"Plot rendering timeout: {plot_status.get('rendered', 0)}/{plot_status.get('total', 0)} plots rendered, continuing anyway...")
            
            # Scroll to ensure all content is rendered
            logger.info("Scrolling to ensure all content is rendered...")
            page.evaluate("() => { window.scrollTo(0, document.body.scrollHeight); }")
            time.sleep(0.5)
            page.evaluate("() => { window.scrollTo(0, 0); }")
            time.sleep(0.5)
            
            # Additional wait for any remaining animations or async operations
            logger.info("Waiting for final render operations...")
            time.sleep(2)
            
            # Wait for window load event (like the live browser does)
            logger.info("Waiting for window load event...")
            page.wait_for_load_state("load")
            time.sleep(1)
            
            # Trigger window resize event to make Plotly recalculate (like browser does)
            logger.info("Triggering window resize to force Plotly recalculation...")
            page.evaluate("""
                () => {
                    // Trigger resize event like browser does
                    window.dispatchEvent(new Event('resize'));
                }
            """)
            time.sleep(0.5)
            
            # Trigger Plotly to resize all plots multiple times (like browser does on load)
            logger.info("Triggering Plotly resize to ensure correct rendering...")
            for i in range(3):
                page.evaluate("""
                    () => {
                        if (window.Plotly && window.Plotly.Plots) {
                          document.querySelectorAll('.js-plotly-plot').forEach(function(el) {
                            try {
                              window.Plotly.Plots.resize(el);
                            } catch (e) {
                              console.warn('Plotly resize failed:', e);
                            }
                          });
                        }
                    }
                """)
                time.sleep(0.3)
            
            # One final resize after a longer wait
            time.sleep(1)
            page.evaluate("""
                () => {
                    if (window.Plotly && window.Plotly.Plots) {
                      document.querySelectorAll('.js-plotly-plot').forEach(function(el) {
                        try {
                          window.Plotly.Plots.resize(el);
                        } catch (e) {}
                      });
                    }
                }
            """)
            time.sleep(0.5)
            
            # Extract HTML and apply transformations - match the original JavaScript code exactly
            logger.info("Extracting and processing HTML...")
            html_content = page.evaluate("""
                () => {
                    // 1. Generate Timestamp for Filename
                    var now = new Date();
                    var offset = now.getTimezoneOffset() * 60000;
                    var localIso = new Date(now.getTime() - offset).toISOString();
                    var timestamp = localIso.slice(2, 19).replace(/[-:]/g, "");
                    var filename = "pixel_patrol_snapshot_" + timestamp + ".html";
                    
                    // 2. Get the full HTML with all computed styles preserved
                    // Use outerHTML which includes all inline styles and computed styles
                    var htmlString = document.documentElement.outerHTML;
                    
                    // 3. Parse DOM (exactly like the original JavaScript code)
                    var parser = new DOMParser();
                    var doc = parser.parseFromString(htmlString, "text/html");
                    
                    // 3. Fix plot heights and ensure widths are preserved
                    doc.querySelectorAll(".js-plotly-plot").forEach((plotDiv) => {
                      const svg = plotDiv.querySelector("svg.main-svg");
                      if (!svg) return;
                      
                      const hAttr = svg.getAttribute("height");
                      const wAttr = svg.getAttribute("width");
                      
                      // Add extra padding for titles and axis labels (typically need 80-120px extra)
                      if (hAttr) {
                        const baseHeight = parseFloat(hAttr);
                        const paddedHeight = baseHeight + Math.min(120, Math.max(60, baseHeight * 0.2));
                        plotDiv.style.height = `${paddedHeight}px`;
                        
                        // Also freeze the immediate parent if present (Dash wraps graphs)
                        const parent = plotDiv.parentElement;
                        if (parent && (parent.style.height === "" || parent.style.height === "auto")) {
                          parent.style.height = `${paddedHeight}px`;
                        }
                      }
                      
                      // Ensure width is preserved from live DOM fixes
                      // If width was set in live DOM, it should be in the style attribute
                      // But also ensure SVG width attribute matches
                      if (wAttr) {
                        const currentWidth = parseFloat(wAttr);
                        // Ensure plot div and SVG maintain the width
                        if (!plotDiv.style.width || plotDiv.style.width === "") {
                          plotDiv.style.width = currentWidth + "px";
                        }
                        if (!svg.style.width || svg.style.width === "") {
                          svg.style.width = currentWidth + "px";
                        }
                        svg.style.maxWidth = "100%";
                        
                        // Also fix all child SVGs
                        plotDiv.querySelectorAll("svg").forEach((childSvg) => {
                          if (childSvg !== svg) {
                            const childWidth = childSvg.getAttribute("width");
                            if (childWidth) {
                              childSvg.style.width = childWidth + "px";
                              childSvg.style.maxWidth = "100%";
                            }
                          }
                        });
                      }
                    });
                    
                    // 4. Set Title
                    doc.title = "PixelPatrol Snapshot " + timestamp;
                    
                    // 4.5. Ensure viewport meta tag exists for proper rendering
                    var viewport = doc.querySelector('meta[name="viewport"]');
                    if (!viewport) {
                        viewport = doc.createElement('meta');
                        viewport.setAttribute('name', 'viewport');
                        viewport.setAttribute('content', 'width=device-width, initial-scale=1.0');
                        doc.head.insertBefore(viewport, doc.head.firstChild);
                    }
                    
                    // 5. CLEANUP: Remove Buttons, tensorboard widget,... from the Snapshot
                    var idsToRemove = [
                        "global-apply-button",
                        "global-reset-button",
                        "export-csv-button",
                        "export-project-button",
                        "save-snapshot-button",
                        "embedding-projector-container"
                    ];
                    
                    idsToRemove.forEach(function(id) {
                        var element = doc.getElementById(id);
                        if (element) {
                            element.remove();
                        }
                    });
                    
                    doc.querySelectorAll(".popover").forEach(function(el) { el.remove(); });
                    
                    // 6. Hide Plotly Modebar
                    var style = doc.createElement('style');
                    style.innerHTML = ".modebar { display: none !important; }";
                    doc.head.appendChild(style);
                    
                    var plotlyFix = doc.createElement("style");
                    plotlyFix.innerHTML = `
                    .js-plotly-plot .svg-container { position: relative !important; }
                    
                    /* Plotly uses multiple sibling SVGs; they must be absolutely overlaid */
                    .js-plotly-plot .svg-container > svg {
                      position: absolute !important;
                      top: 0 !important;
                      left: 0 !important;
                    }
                    
                    /* WebGL layer (if any) */
                    .js-plotly-plot .svg-container > .gl-container {
                      position: absolute !important;
                      top: 0 !important;
                      left: 0 !important;
                    }
                    
                    `;
                    doc.head.appendChild(plotlyFix);
                    
                    // 7. Fix Logo (broken rel links)
                    var logo = doc.querySelector('img[src*="prevalidation"]');
                    if (logo) {
                        var textSpan = doc.createElement("span");
                        textSpan.innerText = "Snapshot";
                        textSpan.style.fontSize = "20px";
                        textSpan.style.fontWeight = "bold";
                        textSpan.style.padding = "0 10px";
                        logo.parentNode.replaceChild(textSpan, logo);
                    }
                    
                    // 8. Force Plotly to relayout/resize after snapshot is opened
                    var script = doc.createElement("script");
                    script.innerHTML = `
                    (function() {
                      function resizeAll() {
                        try {
                          if (!window.Plotly || !window.Plotly.Plots) return;
                          document.querySelectorAll('.js-plotly-plot').forEach(function(el) {
                            window.Plotly.Plots.resize(el);
                          });
                        } catch (e) {}
                      }
                      window.addEventListener('load', function() {
                        resizeAll();
                        setTimeout(resizeAll, 200);
                        setTimeout(resizeAll, 800);
                      });
                    })();
                    `;
                    doc.body.appendChild(script);
                    
                    return doc.documentElement.outerHTML;
                }
            """)
            
            # Save the HTML content
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Verify the file was created
            if not output_path.exists():
                raise RuntimeError(f"HTML file was not created at {output_path}")
            
            logger.info(f"HTML export completed successfully: {output_path}")
            
            browser.close()
    
    finally:
        # Shutdown the server gracefully
        logger.info("Shutting down server...")
        # Since the server thread is daemon=True, it will be killed when main thread exits
        # Give a moment for any final cleanup
        time.sleep(0.5)

