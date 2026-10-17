# 权重

**简体中文** | [English](README.md)

模型文件在运行时放在这里，但**不进 git**。用下面的命令拉取：

```bash
python scripts/fetch_weights.py
```

## 发布了什么

| 架构 | 参数量 | `.pt` | `.onnx` | 角色 |
|---|---:|---:|---:|---|
| `resnet18` | 11.19 M | 44.8 MB | 44.7 MB | 集成成员 1 |
| `efficientnet_b0` | 4.05 M | 16.3 MB | 16.0 MB | 集成成员 2 |
| `mobilenet_v3_small` | 1.53 M | 6.2 MB | 6.1 MB | 备选，默认不用 |
| `resnet34` | 21.30 M | 85.3 MB | 85.2 MB | 备选，默认不用 |

默认组合是 `resnet18 + efficientnet_b0` —— 15.24 M 参数、合计 61 MB，按概率
平均打分。

## 没有随仓库发布的

`hand_yolov8s.pt`（`--full` 用到的手部检测器）是第三方模型，**不在这里再
分发**。用 `--detector` 传它自己的路径。

## 许可证

权重是 CC BY 4.0 —— 见 `../LICENSE-MODEL.md`。
