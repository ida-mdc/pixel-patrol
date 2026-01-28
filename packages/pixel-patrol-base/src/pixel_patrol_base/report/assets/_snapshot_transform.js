/**
 * Shared HTML snapshot transformation logic.
 * Used by both clientside_callbacks.js (browser download) and html_export.py (Playwright CLI export).
 */

/**
 * Transform a parsed HTML document into a clean snapshot.
 * @param {Document} doc - A parsed Document object (from DOMParser)
 * @param {string} timestamp - Timestamp string for the title (e.g., "250128_143022")
 * @param {Object} options - Optional configuration
 * @param {boolean} options.freezeWidths - Whether to freeze plot widths (default: false for browser, true for CLI)
 * @returns {Document} The transformed document
 */
function transformSnapshotHtml(doc, timestamp, options) {
    options = options || {};
    var freezeWidths = options.freezeWidths || false;

    // 1. Fix plot heights and optionally widths
    doc.querySelectorAll(".js-plotly-plot").forEach(function(plotDiv) {
        var svg = plotDiv.querySelector("svg.main-svg");
        if (!svg) return;

        var hAttr = svg.getAttribute("height");
        var wAttr = svg.getAttribute("width");

        // Add extra padding for titles and axis labels (typically need 80-120px extra)
        if (hAttr) {
            var baseHeight = parseFloat(hAttr);
            var paddedHeight = baseHeight + Math.min(120, Math.max(60, baseHeight * 0.2));
            plotDiv.style.height = paddedHeight + "px";

            // Also freeze the immediate parent if present (Dash wraps graphs)
            var parent = plotDiv.parentElement;
            if (parent && (parent.style.height === "" || parent.style.height === "auto")) {
                parent.style.height = paddedHeight + "px";
            }
        }

        // Freeze widths for CLI export (headless browser needs explicit dimensions)
        if (freezeWidths && wAttr) {
            var currentWidth = parseFloat(wAttr);
            if (!plotDiv.style.width || plotDiv.style.width === "") {
                plotDiv.style.width = currentWidth + "px";
            }
            if (!svg.style.width || svg.style.width === "") {
                svg.style.width = currentWidth + "px";
            }
            svg.style.maxWidth = "100%";

            // Also fix all child SVGs
            plotDiv.querySelectorAll("svg").forEach(function(childSvg) {
                if (childSvg !== svg) {
                    var childWidth = childSvg.getAttribute("width");
                    if (childWidth) {
                        childSvg.style.width = childWidth + "px";
                        childSvg.style.maxWidth = "100%";
                    }
                }
            });
        }
    });

    // 2. Set Title
    doc.title = "PixelPatrol Snapshot " + timestamp;

    // 3. Ensure viewport meta tag exists for proper rendering
    var viewport = doc.querySelector('meta[name="viewport"]');
    if (!viewport) {
        viewport = doc.createElement("meta");
        viewport.setAttribute("name", "viewport");
        viewport.setAttribute("content", "width=device-width, initial-scale=1.0");
        doc.head.insertBefore(viewport, doc.head.firstChild);
    }

    // 4. CLEANUP: Remove Buttons, tensorboard widget, etc. from the Snapshot
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

    // 5. Hide Plotly Modebar
    var style = doc.createElement("style");
    style.innerHTML = ".modebar { display: none !important; }";
    doc.head.appendChild(style);

    // 6. Add Plotly positioning fixes for static HTML
    var plotlyFix = doc.createElement("style");
    plotlyFix.innerHTML = [
        ".js-plotly-plot .svg-container { position: relative !important; }",
        "",
        "/* Plotly uses multiple sibling SVGs; they must be absolutely overlaid */",
        ".js-plotly-plot .svg-container > svg {",
        "  position: absolute !important;",
        "  top: 0 !important;",
        "  left: 0 !important;",
        "}",
        "",
        "/* WebGL layer (if any) */",
        ".js-plotly-plot .svg-container > .gl-container {",
        "  position: absolute !important;",
        "  top: 0 !important;",
        "  left: 0 !important;",
        "}"
    ].join("\n");
    doc.head.appendChild(plotlyFix);

    // 7. Fix Logo (broken relative links in static HTML)
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
    script.innerHTML = [
        "(function() {",
        "  function resizeAll() {",
        "    try {",
        "      if (!window.Plotly || !window.Plotly.Plots) return;",
        "      document.querySelectorAll('.js-plotly-plot').forEach(function(el) {",
        "        window.Plotly.Plots.resize(el);",
        "      });",
        "    } catch (e) {}",
        "  }",
        "  window.addEventListener('load', function() {",
        "    resizeAll();",
        "    setTimeout(resizeAll, 200);",
        "    setTimeout(resizeAll, 800);",
        "  });",
        "})();"
    ].join("\n");
    doc.body.appendChild(script);

    return doc;
}

// Export for use in different contexts
if (typeof module !== "undefined" && module.exports) {
    module.exports = { transformSnapshotHtml: transformSnapshotHtml };
}