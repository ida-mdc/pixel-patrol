# Running Pixel Patrol in Containers

Pixel Patrol can run in Docker or Apptainer (Singularity) for reproducible deployments, including Open OnDemand (OOD) on HPC systems.

**For OOD admins**: See [ood/README.md](ood/README.md) for deployment and bind path configuration.

## Path accessibility

The processing dashboard asks for a **base directory** (your data) and **output ZIP path**. These must be paths that exist **inside the container**.

| Runtime | Default bind mounts | What to use |
|---------|---------------------|-------------|
| **Docker** | None | Mount your data: `-v /path/on/host:/data` then use `/data` in the UI |
| **Apptainer** | `$HOME` is bound automatically | Use paths under your home, e.g. `$HOME/projects/imaging` |
| **OOD** | Same as Apptainer | Use `$HOME` or paths your site has bound |

### Docker: mounting paths

Mount the directories you need so they are visible inside the container:

```bash
# Mount your data directory at /data
docker run -p 8051:8051 \
  -v /path/to/your/images:/data \
  pixel-patrol launch

# Then in the UI: Base Directory = /data, Output ZIP = /data/project.zip
```

To match host paths (e.g. use `/home/user` inside the container):

```bash
docker run -p 8051:8051 \
  -v $HOME:$HOME \
  pixel-patrol launch

# Then use your normal paths, e.g. $HOME/projects/imaging
```

### Apptainer: default binds

Apptainer binds `$HOME` by default. Use paths under your home:

- Base directory: `$HOME/projects/imaging` (or wherever your data lives)
- Output ZIP: `$HOME/pixel-patrol-project.zip`

For data outside `$HOME` (e.g. `/scratch`), add a bind:

```bash
apptainer run -B /scratch:/scratch pixel-patrol.sif launch
```

## Building images

### Docker (production, from PyPI)

```bash
docker build -t pixel-patrol .
```

### Docker (development, from local source)

```bash
docker build -f Dockerfile.dev -t pixel-patrol:dev .
```

### Apptainer (from Docker image)

Build the Docker image first, then:

```bash
apptainer build pixel-patrol.sif docker-daemon://pixel-patrol:dev
```

### Apptainer (from definition file, uses PyPI)

```bash
apptainer build pixel-patrol.sif pixel-patrol.def
```

## Running

The launcher app runs on **one port** (8051). It uses `$HOME/pixel-patrol` to store reports and their index. You see a list of existing reports (paths, subpaths, file filters) and can add new ones with a minimal form. Reports open at `/report?zip=...` on the same server. Single-port setup works on clusters.

Set `PIXEL_PATROL_REPORTS_DIR` to override the reports directory (default: `$HOME/pixel-patrol`).

### Docker

```bash
# Launcher + reports (single port 8051, stores in $HOME/pixel-patrol)
docker run -p 8051:8051 -v $HOME:$HOME pixel-patrol launch

# Standalone report viewer (for viewing existing ZIPs without launcher)
docker run -p 8050:8050 -v $HOME:$HOME pixel-patrol report /path/to/project.zip
```

### Apptainer

```bash
# Processing + report (single port)
apptainer run pixel-patrol.sif launch

# Standalone report viewer
apptainer run pixel-patrol.sif report /path/to/project.zip
```

