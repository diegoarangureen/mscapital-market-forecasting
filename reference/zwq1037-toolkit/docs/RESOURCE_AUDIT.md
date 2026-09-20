# 本机资源检查

检查日期：2026-08-22

## 检查结果

| 项目 | 当前情况 | 判断 |
|---|---:|---|
| F 盘剩余空间 | 约 201.4 GB | 足够保存约 10 GB 原始数据及中间文件 |
| 物理内存 | 约 15.3 GB | 可做抽样/分块实验，不适合一次把大表全部装入内存 |
| 检查时可用内存 | 约 5.5 GB | 读取大数据前应关闭不需要的软件 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | 可训练中小型 1D CNN / Transformer |
| GPU 显存 | 约 6.0 GB | 需要控制 batch size、序列长度和模型大小 |
| PyTorch | 2.9.0+cu126 | 已可用 |
| CUDA | 12.6，PyTorch 检测为可用 | 已可用 |

## Python 库状态

已有：`numpy`、`pandas`、`scikit-learn`、`lightgbm`、`torch`、`pyarrow 25.0.1`、`polars 1.43.2`、`psutil 7.2.2`。

第 0 课已经确认这些依赖均可由 `D:\anaconda\envs\pytorch\python.exe` 正常导入。

## 运行策略

这台电脑可以完成项目，但必须采用“大文件放磁盘、按需读小块”的方式：

1. 原始数据放在 `data/raw/`，保持不变。
2. 第一阶段只选少数月份/样本，生成 `data/interim/` 小切片。
3. 先用 CPU 建立 LightGBM 基线。
4. CNN/Transformer 使用 6 GB 显存能承受的小 batch，必要时采用梯度累积。
5. 只有在本地流程正确后，才考虑 Kaggle Notebook 做更大规模训练。
