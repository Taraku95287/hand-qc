# Model weights license

The MIT license in `LICENSE` covers the **source code** in this repository.

The trained weights (`resnet18_handqc.pt`, `efficientnet_b0_handqc.pt`, and the
matching `.onnx` files) are released under **CC BY 4.0**. You may use them
commercially; attribution is required.

## Provenance of the training data

The classifier was trained on 862 hand crops labelled by the author. Those crops
were produced by:

1. Generating illustrations locally with Stable Diffusion checkpoints
   (`rinAnimeBlendArblend_v20`, `waiANIMA_v10Base10`), and
2. Cropping detected hands out of them with `hand_yolov8s`.

The **cropped images are not redistributed** in this repository. Only the
labelling manifest and a small demonstration subset are included, so that the
labelling scheme and the evaluation can be inspected without redistributing a
large derived image set. The generated images themselves follow the terms of the
checkpoints that produced them; the author makes no claim over third-party model
output.

## Third-party components

- `hand_yolov8s.pt` (hand detector) — **not** distributed here. It is a
  third-party model obtained separately; see `docs/DATA_AND_WEIGHTS.md`.
- Architecture definitions come from `torchvision` (BSD-3-Clause).
- `ultralytics` (AGPL-3.0) is only imported when the optional detector is used.
  If you redistribute a product that links against it, review its license.
