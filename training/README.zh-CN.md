# 训练

[![Language](https://img.shields.io/badge/lang-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red.svg)](#)

**简体中文** | [English](README.md)

`train.py` 就是产出交付权重的那个脚本。每次运行训练一个小 CNN —— ImageNet
预训练骨干、单 logit 输出头、带 `pos_weight` 的 `BCEWithLogitsLoss`、AdamW、
余弦学习率、早停、5 折 OOF AUC、可选 ONNX 导出。它是一个**复现用的脚本**，
不是框架：算法与 `docs/RESULTS.md` 背后那次实验完全一致。增强的取值范围一
并保留，包括有意不做水平翻转 —— 镜像会改变这是哪只手，翻转后的样本就对不上
它的标签了。

交付的打分器是两个架构 `resnet18` + `efficientnet_b0`，在概率空间里平均（见
`handqc/backends.py`）。

## 标注清单

`manifest.csv` 是全量标签集：**862 张裁剪图，676 good / 186 bad**。

| 列 | 含义 |
|---|---|
| `label` | `good` 或 `bad` —— 唯一参与训练的列 |
| `hand_id` | 裁剪图标识；图像文件是 `<hand_id>.png` |
| `hand_tag` | 手势，11 个不同取值（`open hand, spread fingers`、`peace sign`、`unknown` 等） |
| `model` | 来源家族：`IL` 518、`Anima` 258、`user` 86（真实出图） |
| `num`、`det_conf` | 行号、裁剪时的手部检测置信度（训练不用） |

没有 `path` 列，因为**裁剪图本身不随仓库发布**。图要你自己提供：

```bash
--image-root /path/to/crops     # must contain every <hand_id>.png from the manifest
```

如果清单里*确实*带 `path` 列，也会被接受；那一列优先，`--image-root` 变成可选。

## 复现交付的模型

默认值就是当时用的设置（`--seed 42 --size 224 --epochs 30 --bs 32 --lr 3e-4
--folds 5 --select best`）：

```bash
python training/train.py --image-root /path/to/crops --arch resnet18
python training/train.py --image-root /path/to/crops --arch efficientnet_b0
cp runs/resnet18_handqc.pt runs/efficientnet_b0_handqc.pt weights/   # names predict.py expects
```

（PowerShell 下用 `Copy-Item`。）`--out` 默认是 `runs/`，里面会得到
`<tag>_handqc.pt`、`<tag>_handqc.onnx`、`oof_<tag>.npz`，以及在给了 `--cross`
时的 `cross_pred_<tag>.json`。`<tag>` 就是 `<arch>`，当 `--size` 不是 224 时加
后缀 `_s<size>`，当 `--select last` 时加 `_last`，这样几次实验不会互相覆盖。

这组设置下记录的 OOF AUC：efficientnet_b0 0.8567、resnet18 0.8330、
mobilenet_v3_small 0.8289、resnet34 0.8107，集成为 0.8664。这些差距大致就是
种子噪声的量级 —— 同一份数据跑两次 OOF AUC 能差约 0.01 —— 所以这个排序不是
结论，也没有被用来挑交付的那一对。

## 运行时间

每个架构大约 10 分钟（224 px 下 5 折 × 30 轮，早停通常会让某一折提前结束），
在开发用的 8 GB 显卡上，四架构整张表约 40 分钟。`--size 320` 大致翻倍。有 CPU
回退路径，但没有做过基准测试。同一张卡上绝对不要同时跑两个 GPU 任务 —— 这个
项目就是这样撞上过静默的驱动重置（TDR）。

## 最要紧的那条注意事项

标签来自单一标注者。对同一批裁剪图做独立盲标注，几乎复现不出任何 good/bad
判断：标为 good 的那些一致性是 100%，标为 bad 的只有 8%；干净子集上 Cohen's
kappa 是 **−0.003** —— 等于零一致。所以从这个脚本得到的 AUC 不是"模型多会
检测解剖错误"，而是"模型多贴近某一个人对'画坏的手'的判定边界"，而这个边界
本身并不自洽。任何在这些标签上训出来的东西都继承这个限制。完整记录见
`docs/RESEARCH_LOG.md` 第 13 与 16 节。
