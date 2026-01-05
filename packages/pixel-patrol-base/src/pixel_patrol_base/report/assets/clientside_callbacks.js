window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        save_snapshot: function(n_clicks) {
            if (n_clicks > 0) {

                // 1. Generate Timestamp for Filename
                var now = new Date();
                var offset = now.getTimezoneOffset() * 60000;
                var localIso = new Date(now.getTime() - offset).toISOString();
                var timestamp = localIso.slice(2, 19).replace(/[-:]/g, "");
                var filename = "pixel_patrol_snapshot_" + timestamp + ".html";

                // 2. Parse DOM
                var parser = new DOMParser();
                var doc = parser.parseFromString(document.documentElement.outerHTML, "text/html");

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
                  // keep width responsive; or uncomment to freeze:
                  // if (wAttr) plotDiv.style.width = `${parseFloat(wAttr)}px`;
                });

                // 3. Set Title
                doc.title = "PixelPatrol Snapshot " + timestamp;

                // 4. CLEANUP: Remove Buttons from the Snapshot
                var idsToRemove = [
                    "global-apply-button",
                    "global-reset-button",
                    "export-csv-button",
                    "export-project-button",
                    "save-snapshot-button"
                ];

                idsToRemove.forEach(function(id) {
                    var element = doc.getElementById(id);
                    if (element) {
                        element.remove();
                    }
                });

                doc.querySelectorAll(".popover").forEach(function(el) { el.remove(); });

                // 5. Hide Plotly Modebar
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

                // 6. Fix Logo (broken rel links)
                var logo = doc.querySelector('img[src*="prevalidation"]');
                if (logo) {
                    var textSpan = doc.createElement("span");
                    textSpan.innerText = "Snapshot";
                    textSpan.style.fontSize = "20px";
                    textSpan.style.fontWeight = "bold";
                    textSpan.style.padding = "0 10px";
                    logo.parentNode.replaceChild(textSpan, logo);
                }

                // --- 7. Force Plotly to relayout/resize after snapshot is opened ---
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


                // 8. Download
                var htmlContent = doc.documentElement.outerHTML;
                var blob = new Blob([htmlContent], {type: "text/html"});
                var url = URL.createObjectURL(blob);

                var a = document.createElement("a");
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url); // Clean up memory
            }
            return window.dash_clientside.no_update;
        }
    }
});