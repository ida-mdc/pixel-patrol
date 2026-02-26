# Pixel Patrol OOD App

Deploy: `cp -r . /var/www/ood/apps/sys/pixel_patrol` (or `~/ondemand/dev/pixel_patrol` for dev).

Configure in `submit.yml.erb`:
- `PIXEL_PATROL_SIF_PATH` – path to SIF image
- `PIXEL_PATROL_BIND_PATHS` – cluster filesystems, e.g. `"/scratch:/scratch:ro,/projects:/projects:ro"`
- `form.yml` – partitions, walltime

Two bindings: input (base directory) and output (ZIP path). `$HOME` bound automatically; add `PIXEL_PATROL_BIND_PATHS` for other input locations. CPU-only (no GPU).
