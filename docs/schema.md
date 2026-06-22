# Report Schema

Every Pixel Patrol report is a single `.parquet` file. Its columns are assembled
from the registered **loaders** (image metadata), **processors** (metrics), and a
set of pipeline-generated columns; the **viewer widgets** then consume those
columns to draw the report.

The same per-column descriptions below are embedded into each report's parquet
field metadata, so produced files are self-describing.

Two companion resources are generated directly from the installed plugins, so
they always match the build:

- **[Interactive report map](assets/schema.html)** — follow any *producer → column
  → widget* path. Click a node for its description and connections; hover to trace
  its links.
- **[Full catalog as JSON](assets/schema.json)** — the machine-readable schema and
  plugin catalog. Also available from the CLI:

  ```bash
  pixel-patrol schema -o schema.json
  ```

## All report columns

Every column that can appear in a report, with the source that produces it. The
table is generated from the installed plugins.

<!-- BEGIN GENERATED COLUMNS (docs/gen_schema_docs.py) -->
<!-- END GENERATED COLUMNS -->
