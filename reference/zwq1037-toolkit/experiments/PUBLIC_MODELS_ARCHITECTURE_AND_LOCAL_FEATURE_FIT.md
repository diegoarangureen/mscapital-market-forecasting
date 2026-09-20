# 当前 0.149 融合中的公开模型：结构与本地特征适配

日期：2026-09-14。此文只读代码并整理，不训练、不复现、不提交。这里的“可适配”指可用我们的本地特征训练**新模型**，不表示能够重现公开模型的 LB。

| 公开来源及融合权重 | 实际结构与输入 | 接入本地 379 列的可行性 |
|---|---|---|
| bestwater GPU TabM，20% | 689 列表格输入＝462 列作者特征 + 152 列公开特征 + 75 列横截面特征。共享 MLP：512 宽、3 个隐藏块、末层 256；64 个线性头先平均，再以 raw target 的 batch cosine loss 训练；3 seed × 5 时间折。 | **可直接换数值输入维数**，但作者的 462 列未随 Notebook 公开，无法用我们的 379 列复现原 0.142。由于 64 个线性头在 loss 前平均，数学上等价于一个线性输出头，区别主要是共享 MLP、cosine loss、特征和多折训练。 |
| bestwater TPU TabM，12% | 与 GPU 版同一共享 MLP 配置：64 头、512 宽、3 块、dropout 0.15、同三个 seed；TPU 实现的 batch 和训练细节不同。两份现成测试预测相关性约 0.986。 | **无需另做一种架构适配**；如果以后试这一思路，先在 GPU 上做一版即可。 |
| YangQ 公开融合，32% | 内部为 60% 五个 MultiStream CNN–Transformer 成员 + 40% 八成员 RealMLP。前者分别编码 market、transaction、order 时间网格，再跨流混合；后者对表格数值做 PBLD 周期嵌入、成员独立线性层、256→256→64 回归。 | **RealMLP 分支最容易改用 379 列**；只需换输入矩阵并按训练折拟合其数值缩放。序列分支不能直接吃 379 列；若加入静态特征须改网络。精简包没有原训练网格 v2/v3 与私有 factor 表，不能原样重训公开 0.142。 |
| Yunsu RealMLP，5% | 实际训练为 16 成员数值周期嵌入 + 类别输入 + 512→512→128 MLP；除了回归头还有 RQ-KMeans 目标编码辅助头，训练含加权 MSE、cosine 和辅助分类损失。原 Notebook 读取 `rfmf-0726data`。 | **可改成 379 列数值输入，但适配量较大**：当前实现假设类别分支与 RQ 辅助任务都存在。先做低成本的数值分支架构筛选，再决定是否复现完整 16 成员及辅助损失。 |

我们自己的 Hybrid Transformer 占 16%，内部旧融合占 15%；它们不是上表的公开模型。当前 379 列双种子 TabM 没有进入 0.149 方案。

若将来开展本地前向验证，优先顺序是：① YangQ v10 数值 RealMLP + 我们 379 列；② bestwater 共享 MLP + 同一套 379 列；③ 只有前两项显示增益时，才考虑给 MultiStream 序列模型新增静态特征支路。这个顺序基于接入难度和结构互补性，不是已测出的分数排序。

代码来源：`data/external/kaggle_public/bestwater_cos689/kgpu-tabm-cos689-3seed.py`、`data/interim/public_notebooks/tabm_ktpu/ktpu-tabm-cos-v6-3seed.py`、`data/interim/public_notebooks/lb0142_pack/models_v9.py`、`models_v10.py`、`train_v10.py`、`data/interim/public_notebooks/realmlp/rfmf-realmlp.ipynb`。公开融合比例与来源见 `outputs/submission_metadata/robust6_knownlb_equal38_common.json`。

