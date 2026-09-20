# 公开 TabM 与 Transformer 代码对比

日期：2026-09-13

## 公开 TabM

来源：`bestwater/kgpu-tabm-cos689-3seed` 与对应 TPU 版本。

- 输入为 689 个表格特征：462 个自制特征、152 个公开特征，以及 75 个横截面特征。
- 75 个横截面特征由 30 个同月 percentile rank、15 个同月 z-score、30 个样本内 cross-feature rank 组成。
- 网络是共享 MLP 主干加 64 个线性输出头，最终先平均 64 个头再计算 loss。由于线性头的平均仍是一个线性层，这个实现没有官方 TabM 的独立成员特征变换，本质上是普通深 MLP。
- 使用 raw target、batch 2048 的中心化 cosine loss、3 个 seed、5 个带两个月间隔的时间折和 early stopping；测试预测平均 15 个模型。
- 公开讨论报告该方案单独提交约 0.142；全量无 holdout 重训下降到约 0.137。

需要谨慎解释它的 CV：用于产生横截面特征的 top-30 特征由全训练集标签监督选择；所有折又统一使用 `month <= 62` 的均值和标准差，因此早期验证折含有未来信息。测试集的同月 rank/z-score 实际也把整个测试集当成一个月。

我们已公平测试过的部分：Relative319 official TabM 上的纯 cosine、MSE+cosine、periodic embedding、同月 rank+z-score。纯 cosine 与混合 loss 没有跨窗口/seed 稳定复现；rank+z-score 只在很晚月份改善，完整 62–70 窗口略降。因此公开方案中尚未验证的核心不是 loss，而是更大的 462+152 特征并集与多时间折模型平均。

## 公开 Transformer baseline

来源：`sweetyheehee/transformer-baseline`。

- 不直接使用 319 个聚合特征，而是把三张原始表分别压成固定时间网格：market 为 200×11（覆盖 600 秒），transaction 为 60×7（覆盖 60 秒），order 为 60×10（覆盖 60 秒）。
- 三路各自线性投影，并添加位置 embedding 和数据表类型 embedding；随后拼成 320 个 token。
- Transformer 前使用 kernel 5 与 kernel 3 的残差 Conv1D，主体为 2 层、4 头、`d_model=96`，最后用 attention pooling 回归。
- padding mask 完整；归一化只从训练部分抽样拟合。
- 默认只随机使用最多 40 万训练样本，训练月 `<56`、验证月 `>=56`，5 epoch。
- loss 为 35% SmoothL1 + 65% batch cosine；按验证 cosine 保存最佳 checkpoint。

这与我们的 Joint-Transformer 有本质差别。我们的序列只来自 market 逐事件表（其中带成交字段），长度 200、22 个通道，再与 Relative319 静态分支结合；没有 transaction.feather 和 order.feather 的独立时间序列。我们的网络在线性投影后直接进入 Transformer，没有卷积前端，并使用 MSE。

## 公开 0.142 MultiStream 包

该包的序列模型进一步为 market、transaction、order 各建一个 CNN–Transformer encoder，再用一层 Transformer 在三个 stream 表示之间做 cross-stream mixing。最终 0.142 配方不是单一 Transformer：它是 5 个不同长度/容量的 MultiStream 模型平均后占 60%，另一个 8-member RealMLP 占 40%。训练脚本使用全量数据、纯 cosine loss，并带测试集 BatchNorm adaptation 与预测去均值；精简包缺失 `online.py` 和网格生成部分，无法完整复现或判断每个单模分数。

## 对下一步的判断

最值得测试的是 `MultiStream-lite Joint-Transformer`：保留我们的 Relative319 静态分支、MSE、时间切分、seed 与训练预算，只把当前 market-only 序列编码器替换为公开 baseline 的三路固定网格 + Conv1D + Transformer。这样主要检验“独立 transaction/order 时序是否带来增量”，也避免再次测试已经失败的 cosine loss。

若这一版在 50–59 晋级，再固定配置跑 60–70；若失败，再考虑 TabM 的多时间折模型平均。公开 TabM 的 64 heads 不值得移植，因为它们在数学上等价于一个线性输出头。

