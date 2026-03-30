# pixel-patrol-tensorboard

Small CLI for opening TensorBoard Embedding Projector from Pixel Patrol data.

## Install

```bash
uv pip install -e packages/pixel-patrol-tensorboard
```

## Usage

```bash
pixel-patrol-tensorboard SOURCE [--port 6006]
```

`SOURCE` can be:
* `.parquet`
* `.arrow` / `.ipc`
* `.zip` Pixel Patrol export (requires `pixel-patrol-base`)

### Example

```bash
pixel-patrol-tensorboard examples/out/quickstart.parquet
```

Then open:

```
http://127.0.0.1:6006/#projector
```

Press `Ctrl+C` to stop.