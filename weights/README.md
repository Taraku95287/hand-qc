# Weights

**English** | [简体中文](README.zh-CN.md)

The model files live here at runtime but are **not** stored in git. Fetch them
with:

```bash
python scripts/fetch_weights.py
```

## What is shipped

| architecture | params | `.pt` | `.onnx` | role |
|---|---:|---:|---:|---|
| `resnet18` | 11.19 M | 44.8 MB | 44.7 MB | ensemble member 1 |
| `efficientnet_b0` | 4.05 M | 16.3 MB | 16.0 MB | ensemble member 2 |
| `mobilenet_v3_small` | 1.53 M | 6.2 MB | 6.1 MB | alternative, not used by default |
| `resnet34` | 21.30 M | 85.3 MB | 85.2 MB | alternative, not used by default |

The default pair is `resnet18 + efficientnet_b0` — 15.24 M parameters,
61 MB combined, scored with a probability average.

## Not bundled

`hand_yolov8s.pt` (the hand detector used by `--full`) is a third-party model
and is **not** redistributed here. Pass its path with `--detector`.

## License

Weights are CC BY 4.0 — see `../LICENSE-MODEL.md`.
