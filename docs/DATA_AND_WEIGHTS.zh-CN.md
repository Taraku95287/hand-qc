# 数据与权重的来源

**简体中文** | [English](DATA_AND_WEIGHTS.md)

## 仓库里有什么

| 产物 | 是否随仓库发布 | 说明 |
|---|---|---|
| `training/manifest.csv` | ✅ 862 行 | `num, label, hand_id, hand_tag, model, det_conf` —— 不含图像路径，不含像素 |
| `training/oof_<arch>.npz` | ✅ 4 个文件，每个 5 KB | OOF 的 `P(good)`、标签、折号、种子、选择模式 |
| `training/train.py` | ✅ | 训练脚本，路径已外置 |
| `examples/*.png` | ✅ 8 张裁剪图 | 4 张 good / 4 张 bad，供快速上手用 |
| `weights/*.pt`、`weights/*.onnx` | GitHub Release | 发布的那一对共 61 MB；由 `scripts/fetch_weights.py` 拉取 |
| **手部裁剪图（862 张）** | ❌ | 不再分发 |
| **原始插画** | ❌ | 不再分发 |
| **`hand_yolov8s.pt`** | ❌ | 第三方模型，不由我分发 |

manifest 与 OOF 向量是特意一起发布的：有了它们，`RESULTS.zh-CN.md` 里每张表都能复现（`python scripts/evaluate.py`），而不必把一套派生图像集散出去。

## 训练数据是怎么做出来的

1. **生成。** 插画在本地用 Stable Diffusion 底模生成，每种手势用固定提示词模板，让手势、构图和取景在整个数据集里大致恒定：
   - `rinAnimeBlendArblend_v20` —— Illustrious 家族，518 张
   - `waiANIMA_v10Base10` —— Anima，258 张
   - 另外 86 张是从已发布的真实画作里裁出来的，用来把真实分布带进训练（这就是 `RESULTS.zh-CN.md` 里写的那个修复，29.8% → 2.1% 误报）
2. **检测。** `hand_yolov8s` 定位手；框扩大 1.2×，最短边 48 px，采集时检测置信度 ≥ 0.25。
3. **标注。** 每张裁剪图由一位标注者标成 `good` 或 `bad`，同时记下手势标签和底模标签。手势标签都核对过确实存在于 Danbooru 标签表里，好让提示词的用词与这些模型训练时响应的词对上。
4. **分折。** 对 `model|hand_tag` 字符串做 `StratifiedKFold(5)`，保证没有哪一折里的底模或手势是别的折没见过的。这样堵掉了两条本来会把所有数字抬高的捷径：「Illustrious 的手都是坏的」和「clenched hand 都是坏的」。

## 为什么不分发裁剪图

两条原因，按重要性排序：

1. **它们是第三方模型输出的派生品。** 裁剪图是具名 Stable Diffusion 底模生成图像的碎片。条款归那些底模所有，重新托管一套派生图像集不是作者能决定的事。来自已发布真实画作的那 86 张裁剪图，更是明确不可再分发。
2. **不需要。** 所有定量结果都是 manifest 加 OOF 向量的函数，这两样都已发布。

`examples/` 里有 8 张裁剪图，纯粹是为了让读者验证新装的这套能跑。它们的标签在 `examples/labels.csv` 里。

## 手部检测器

`--full` 模式需要一个 YOLOv8 手部检测器。它**没有**包含在仓库里：

- 它是第三方模型。
- 它的运行时 `ultralytics` 是 **AGPL-3.0**。在这个仓库里 import 它没问题；但把链接它的应用再分发出去是个许可决定，应该主动做，而不是无意中继承。

把 `--detector` 指向你自己那份。除 `--full` 之外的功能都不需要它：`--image` 和 `--dir` 直接收裁剪图。

**已知盲区，实测：** 对整张图里唯一的手就是握拳的图像，`hand_yolov8s` 漏掉 **82%**。它的训练数据里这类样本显然极少。如果报告写着「找到 0 只手」，那不能当作手没问题的证据。

## 权重

| 架构 | 参数量 | `.pt` | `.onnx` |
|---|---:|---:|---:|
| `resnet18` | 11.19 M | 44.8 MB | 44.7 MB |
| `efficientnet_b0` | 4.05 M | 16.3 MB | 16.0 MB |
| `mobilenet_v3_small` | 1.53 M | 6.2 MB | 6.1 MB |
| `resnet34` | 21.30 M | 85.3 MB | 85.2 MB |

前两个是发布的集成。后两个是可选项，默认不用：OOF AUC 分别是 0.8289 和 0.8107，都低于发布的两个成员中的任何一个。

ONNX 导出用的是 `torch.onnx.export(..., dynamo=False, opset_version=14)`。在 8 张示例裁剪图和全集上都与 PyTorch 路径对过：最大绝对误差 **8.9e-07**（CPU）。在 CUDA 上 PyTorch 路径用 TF32 卷积，会把误差放大到 ~8.5e-04 —— 那是硬件的默认精度，不是导出出错。

## 复现训练

```bash
cd training
python train.py --manifest manifest.csv --image-root /path/to/your/crops \
    --arch resnet18 --seed 42
```

你需要裁剪图，而这些不随仓库发布。目录结构和关于标签的诚实提醒见 `training/README.md`。
