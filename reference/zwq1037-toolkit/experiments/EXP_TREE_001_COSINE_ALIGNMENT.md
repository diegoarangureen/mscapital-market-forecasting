# EXP-TREE-001：Cosine 对齐路线小样本研究

## 目的与固定边界

- 日期：2026-09-09
- 特征：EXP-002 的62个 market 聚合特征；不含 `sample_id`、`month`、`target` 或其派生列。
- 时间划分：训练月份0～59，验证月份60～70；从未随机划分。
- 方法：两个互不重叠的确定性筛选样本，分别取 `sample_id % 10 == 0` 与 `== 1`。每个样本都有106,417条训练行和19,347条验证行。
- 产物：`data/interim/tree_cosine_research/sample_mod_10_remainder_0/` 与 `sample_mod_10_remainder_1/`，各自保存配置、OOF预测、验证预测、总体和逐月cosine及模型文本。

## 关键数学边界

官方指标是整组向量的

$$
\operatorname{cos}(y,p)=\frac{y\cdot p}{\lVert y\rVert\lVert p\rVert}。
$$

因此它不是逐样本可加的损失。对任意正数 $a$，有
$\operatorname{cos}(y,ap)=\operatorname{cos}(y,p)$；实验中 `p` 与 `7p` 的分数逐位一致。这说明只做预测缩放不会优化官方指标。

若直接最小化 $-\operatorname{cos}(y,p)$，每个预测的梯度依赖整组预测的范数；完整Hessian还含不同样本之间的非对角耦合项。LightGBM自定义objective接口只接收逐行gradient/hessian，无法表达这些交叉项。我们实现了精确梯度和裁剪后的Hessian对角线，仅作为接口诊断，不能称为“LightGBM 精确 cosine loss”。

## 结果

| 路线 | remainder 0 | remainder 1 | 结论 |
|---|---:|---:|---|
| L2 原始target | 0.080826 | 0.053295 | 对照 |
| 预测自身去均值 | 0.082321 | 0.053879 | 无标签后处理有小提升，但会依赖整批预测分布 |
| 训练target去均值且不加回 | 0.082268 | 0.054176 | 两个互斥样本均提升；可进入全量验证 |
| 时间顺序OOF常数平移 | 0.081759 | 0.053306 | 不稳定、接近无增益 |
| L2 + `abs(target)`样本权重 | 0.061910 | 0.040429 | 明显退步，不保留 |
| 自定义负cosine对角近似 | 0.043236（第1轮） | 0.045580（第1轮） | 可运行但不可靠，不进入全量 |

验证target参与拟合的“最优常数平移”只作为上界诊断，分别为0.082726和0.055771；绝不作为有效验证分数。

## 决策

进入 EXP-TREE-002：只检验训练期target去均值；保持EXP-002的特征、时间边界、LightGBM参数和cosine early stopping 不变。没有采用伪逐样本 cosine objective、验证target校准或目标幅度样本权重。
