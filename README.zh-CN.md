# hand-qc —— 筛查二次元插画里画坏的手

[![Language](https://img.shields.io/badge/lang-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**简体中文** | [English](README.md)

一个给**手部裁剪图**打分的小分类器，只回答一个问题：*这只手看起来画对了吗？*
它是为 AI 出图的发布前筛选做的，因为在各类缺陷里，画坏的手出现得最频繁。

- **61 MB**，15.2 M 参数，不用 CLIP，不用基础大模型，CPU 即可运行。
- 在 862 张人工标注的手部裁剪图上取得 **OOF AUC 0.8664**；交叉验证的折按
  *手势 × 底模* 分层，两者都无法被记住。
- 同时发布 **PyTorch 与 ONNX** 权重，以及 **out-of-fold 预测向量**，所以
  `docs/RESULTS.md` 里每一个数字都能在不接触图片数据的情况下复算。

```bash
python scripts/fetch_weights.py
python predict.py --image examples/good_02.png      # -> score 0.9982
python predict.py --image examples/bad_03.png       # -> score 0.1371
```

---

## 这个工具到底在做什么

给它一张手部裁剪图，它返回 `P(good)`，这里的"好"指*人不会把这只手标出来*。
它在作者亲手标注的 862 张裁剪图上训练，复现的就是这个判断，仅此而已。

**它不是"手部解剖结构正确性检测器"。** 有两组测量把这句话逼进了 README：

1. 对作者自己的 `bad` 标签做盲测三级复检：只有 **2%** 能被复现成客观的结构
   错误；**81%** 在第二遍被判为没问题。干净子集上的一致性是
   **Cohen's κ ≈ 0（−0.003）**。
2. 对真实用户作品重新标注后，客观错误率只有 **约 2%** —— 远低于"AI 画不好
   手"这种说法会给出的预期。

所以诚实的说法是：这是**某一个人审美判断的廉价且稳定的复现**，适合当分诊
过滤器；而这个项目真正有意思的地方，最后变成了*测量这个任务本身有多含糊*，
而不是把 AUC 往上刷。值得一读的五条结论在 `docs/FINDINGS.md`。

## 结果

集成 = `resnet18 + efficientnet_b0`，概率平均。`bad` 是正类：分数**低**就是报警。

| 指标 | 数值 |
|---|---|
| out-of-fold AUC，862 张裁剪图 | **0.8664** |
| 阈值 0.50 处的准确率 | 83.2% |
| 0.50 处 `bad` 的精确率 / 召回率 | 59.6% / 68.3% |
| 0.547 处 `bad` 的精确率 / 召回率（对齐到 15% 误报） | 56.4% / 71.0% |
| 什么都不做的基线（全部判为好） | 准确率 78.4%，抓出 0 个坏手 |
| 在**真实用户作品**上重标定后的误报率 | **2.1%** |
| 模型体积 / 参数量 | 61 MB / 15.24 M |
| 打分吞吐 | 每只手约 9 ms |

单手势 AUC 从 **0.68**（`clenched hand` 握拳）跨到 **0.92**（`own hands
together` 双手相握），跨度 0.24 —— 这个差异被单个头条数字盖住了。完整表格、
跨模型评测和消融实验都在 **`docs/RESULTS.md`**；`python scripts/evaluate.py`
可以只用仓库里发布的 OOF 向量重算核心表格。

### 赢过和赢不过的基线

| 方案 | 体积 | OOF AUC |
|---|---|---|
| CLIP ViT-H/14 线性探测 ¹ | 2411 MB | 0.7709 |
| CLIP ViT-H/14 微调 ¹ | 2411 MB | 0.7957 |
| 单用 resnet18 | 45 MB | 0.8330 |
| 单用 efficientnet_b0 | 16 MB | 0.8567 |
| **交付的集成** | **61 MB** | **0.8664** |
| 三模型集成 | 67 MB | 0.8706 |

¹ 在更早的 776 张旧集上测得；CLIP 那两组没有在最终 862 张上重跑。小模型各行
都是 862 张。

从 0.8330 抬到 0.8664 靠的是集成，不是挑架构：四个骨干的单跑分数（0.811–0.857）
差距大致等于种子噪声，按单次交叉验证分数挑一个，等同于掷硬币。

## 安装

```bash
pip install -r requirements.txt          # numpy, opencv, torch, torchvision
python scripts/fetch_weights.py          # 61 MB into ./weights/
```

只有 CPU、没有 CUDA 也能用：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

完全不想装 torch？走 ONNX 这条路：

```bash
pip install numpy opencv-python onnxruntime
python scripts/fetch_weights.py --onnx-only
python predict.py --image examples/good_02.png --backend onnx
```

在 862 张裁剪图上，ONNX 与 PyTorch 的最大绝对误差是 **8.9e-07**。

## 用法

```bash
# one crop
python predict.py --image examples/good_02.png

# a folder
python predict.py --dir examples/ --json out/scores.json

# a whole illustration: detect hands first, then score each
python predict.py --full art.png --detector weights/hand_yolov8s.pt
```

`--full` 需要一个 YOLOv8 手部检测器，那是**第三方模型，没有随仓库发布**
（`ultralytics` 是 AGPL-3.0，见 `LICENSE-MODEL.md`）。没有检测器就自己把手
裁出来，然后用 `--image` / `--dir`。

### 用之前必须重定阈值

在训练域里标定出来的阈值，用到真实作品的好手上，误报率是 **29.8%** —— 十张
好图里有三张被报警。往训练集里补 86 张真实分布的裁剪图之后降到 **2.1%**；
但只要你换底模、换画风或换渲染流程，分数分布就可能再次平移。

`calibrate.py` 在**你自己的图上、不需要任何标注**重定阈值：它假设你大部分手
是好的（客观错误率约 2%），然后取那个能拒掉 `--target-fpr` 比例的分位数。

```bash
python calibrate.py --dir my_renders/ --detector weights/hand_yolov8s.pt --target-fpr 0.10
python predict.py --full new_art.png --detector weights/hand_yolov8s.pt --threshold 0.412
```

## 工作原理

```
illustration ──YOLOv8 hand detector──► crops (bbox ×1.2, min 48 px, det conf ≥ 0.40)
                                          │
                          resnet18 ───────┤
                       efficientnet_b0 ───┤──► probability average ──► P(good) ──► threshold
```

有两个设计点很容易做错：

- **预处理用 letterbox，不拉伸。** 手本来就不是正方形，把裁剪图压成正方形会
  改变模型正在判断的手指长度比例。做法是长边缩到 224 px，短边居中补边。
- **数据增强里没有水平翻转。** 其余不涉及镜像的增强全都用了（旋转 ±12°、缩放、
  HSV、对比度、15% 概率转灰度），但把一只手镜像会把它从左撇子变成右撇子，
  等于教模型"左右手是噪声"。

检测器的置信度还被当作免费的分布外闸门用。在 480 张标注过的裁剪图（其中 168
张确认为非手）上，直接用检测置信度区分"手 / 不是手"，**AUC 0.9117**；同样这批
数据训一个 CNN 闸门只有 0.7532，换到更大更难的一组掉到 0.6873。第三方插画上
约 40% 的检测框根本不是手 —— 灯笼、发辫、兽爪、花、树 —— 所以这层闸门省不掉。

## 排错

- **`FileNotFoundError: missing weights`** —— 先跑
  `python scripts/fetch_weights.py`。默认权重目录是*与 `predict.py` 同级*的
  `./weights`，不是当前工作目录，所以用绝对路径调用脚本也能找到。
- **`fetch_weights.py` 报连接错误** —— 如果你在代理后面，设置
  `HTTPS_PROXY`/`HTTP_PROXY`，或者从 Release 页面手动把四个文件放进 `weights/`。
- **ONNX 打印 `Failed to create CUDAExecutionProvider`** —— 无害。CUDA 版
  onnxruntime 在没有 CUDA 运行库的机器上会列出这个 provider，然后加载失败；
  `OnnxBackend` 会回退到 CPU 并在输出里说明。
- **明明有手却 `0 hands found`** —— 见下面的局限：检测器漏掉了绝大多数握拳图。

## 已知局限

- **检测器看不见的，它也就看不见。** `hand_yolov8s` 会漏掉 **82%** 只有握拳
  手部的图像。"检出 0 只手"不能当作"手没问题"的证据。
- **恰恰在人们会争的地方最弱。** 握拳（AUC 0.682）和持物（0.748）是最差的两个
  手势，而这两个手势正是边界最含糊的。
- **只有一个标注者，没有标注者间一致性。** 标签都是作者一个人的。引用任何准确
  率之前，先看 `docs/FINDINGS.md` 第 1 条。
- **数据集小且窄。** 862 张裁剪图来自两个 SD 底模，另加 86 张真实用户作品。
  `bad` 类样本不足（186 张），而好画的手势上 `good` 类又过度集中。
- **阈值是按分布走的。** 换模型或换画风就要重跑 `calibrate.py`。
- **它不是通用的质量指标。** 它只给手打分，不知道这张图整体好不好看。

## 仓库结构

```
predict.py              CLI: score crops, or whole illustrations
calibrate.py            CLI: fit a threshold on your own unlabelled images
handqc/                 importable package
  preprocess.py         unicode-safe I/O, letterbox, normalization
  backends.py           PyTorch and ONNX scorers, single API
  detector.py           optional YOLOv8 hand detector + OOD gate
  calibration.py        quantile thresholding, no labels required
scripts/
  fetch_weights.py      pull the 61 MB of weights from the release
  evaluate.py           rebuild the RESULTS.md tables from shipped OOF vectors
  check_docs.py         verify the English/Chinese doc pairs have not drifted
training/
  train.py              reproduction of the training run
  manifest.csv          862 rows: label, gesture, base model, detector conf
  oof_*.npz             out-of-fold predictions + fold + seed
docs/
  RESULTS.md            every measured number, including the failures
  FINDINGS.md           the five results worth reading
  RESEARCH_LOG.md       the raw chronological log (1.6k lines, has retractions)
  DATA_AND_WEIGHTS.md   where the data came from and what is not redistributed
MODEL_CARD.md           intended use, out-of-scope use, ethical notes
```

## 许可证

代码 MIT（`LICENSE`）。权重 CC BY 4.0（`LICENSE-MODEL.md`）。
`hand_yolov8s` 是第三方模型，不随本仓库发布。训练用的裁剪图也不发布 —— 见
`docs/DATA_AND_WEIGHTS.md`。
