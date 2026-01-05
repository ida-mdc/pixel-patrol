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