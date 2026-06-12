# Pixel Patrol Launcher

A self-installing double-click binary for Pixel Patrol.

## What it does

| Run | Behaviour |
|-----|-----------|
| First launch | Downloads [uv](https://github.com/astral-sh/uv), creates a managed Python environment at `~/.pixel-patrol/venv/`, installs the `pixel-patrol` package with all its dependencies, then opens the Pixel Patrol web UI in your browser. |
| Subsequent launches | Opens the Pixel Patrol web UI directly. |

The environment lives in `~/.pixel-patrol/`.
To reset: delete that directory and launch again.

## Building the binary

Requires the `pixel-patrol` micromamba environment.

```bash
cd deploy/launcher
./build.sh
# → dist/pixel-patrol-launcher
```

The resulting binary is self-contained (~25 MB) and requires no Python
installation on the target machine.  It downloads `uv` (~10 MB) on first
launch to manage the pixel-patrol environment.

## Making it double-clickable on Linux (GNOME / KDE)

Most file managers will prompt "Run in Terminal" when you double-click the
binary.  For a smoother experience, create a desktop entry:

```bash
cat > ~/.local/share/applications/pixel-patrol.desktop << EOF
[Desktop Entry]
Name=Pixel Patrol
Comment=Image quality inspection tool
Exec=/path/to/pixel-patrol-launcher
Icon=utilities-system-monitor
Terminal=false
Type=Application
Categories=Science;Graphics;
EOF
```

Replace `/path/to/pixel-patrol-launcher` with the actual path.
