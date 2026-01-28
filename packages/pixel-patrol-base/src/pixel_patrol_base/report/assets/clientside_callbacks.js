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

                // 3. Apply shared transformations (freezeWidths=false for responsive browser export)
                if (typeof transformSnapshotHtml === "function") {
                    transformSnapshotHtml(doc, timestamp, { freezeWidths: false });
                } else {
                    console.error("_snapshot_transform.js not loaded - snapshot may be incomplete");
                }

                // 4. Download
                var htmlContent = doc.documentElement.outerHTML;
                var blob = new Blob([htmlContent], {type: "text/html"});
                var url = URL.createObjectURL(blob);

                var a = document.createElement("a");
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
            return window.dash_clientside.no_update;
        }
    }
});