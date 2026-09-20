# 第 6C 课：完整数据 CNN 实验（EXP-CNN-001）

## 1. 本课目标

这一课不再发明新的数据结构，而是把已经验收的四部分接成一次正式实验：

```text
6.56 GiB序列缓存
→ MemmapSequenceDataset按需读取
→ BlockShuffleBatchSampler块级打乱
→ mask-aware 1D CNN
→ month 0～59训练
→ month 60～70验证
→ 保存并重新载入最佳模型
→ 与LightGBM EXP-002公平比较
```

学完后，你应该能够解释：

1. 为什么训练批次可以打乱，而单个样本内部的200个时间位置不能打乱；
2. `batch_sampler`怎样同时兼顾训练随机性和连续磁盘读取；
3. 为什么最终比较的是“最佳验证轮”，而不是机械使用最后一轮；
4. 保存的权重怎样装回同结构的新模型。

本课不做网络调参、残差连接、多数据流融合或Kaggle提交。先得到可信的第一版完整CNN基线，再根据证据决定下一次只改哪个变量。

## 2. 配套阅读

资料：《动手学深度学习》第二版，6.6.2“模型训练”。

- 本地书本页码：243～245页；PDF页码：261～263页。
- 时机：运行正式训练前浏览10分钟。
- 深度：复习训练模式、GPU batch、按epoch评估的整体流程，不需要逐行抄代码。
- 暂时跳过：Fashion-MNIST分类准确率、`d2l.Accumulator`、动画绘图、Xavier初始化以及书中的SGD参数选择；我们的任务是金融回归，使用MSE、Adam和余弦验证。

读后思考：

1. 为什么训练时要调用`model.train()`，验证时要调用`model.eval()`？
2. 为什么模型在GPU上时，每个batch也必须移动到GPU？
3. 为什么训练loss持续下降，仍不能代替验证集余弦分数？

这部分是复习而不是新的阅读负担。数据缓存、mask和block shuffle属于本项目特有实现，由本讲义完整解释。

## 3. 这次到底在比较什么

LightGBM EXP-002使用的是每个样本一行的62个聚合特征。CNN使用的是每个样本最多200个时刻的14通道序列：

```text
LightGBM输入：(sample, 62个聚合数)
CNN输入：     (sample, 14个通道, 200个时间位置)
```

两者使用相同的训练月份、验证月份和余弦指标，因此分数可以比较。但如果CNN分数不同，不能简单断言差异只来自“CNN比LightGBM好或坏”，因为输入表示也不同：

- LightGBM看人工聚合后的统计特征；
- CNN看更接近原始形态的market序列；
- 当前CNN只看market，不看order和transaction独立数据流。

这次实验回答的是：

> 第一版market序列CNN，在可信时间验证上能否学到超过聚合MLP、并接近或超过LightGBM的信号？

## 4. 正式数据规模

固定时间划分为：

| 部分 | 月份 | 样本数 | batch size 512时的batch数 |
|---|---:|---:|---:|
| 训练集 | 0～59 | 1,064,163 | 2,079 |
| 验证集 | 60～70 | 193,474 | 378 |

最后一个训练batch只有227个样本，因为：

$$
1{,}064{,}163 = 2{,}078\times512 + 227
$$

最后一个验证batch有450个样本：

$$
193{,}474 = 377\times512 + 450
$$

因此训练循环不能假设每一批永远是512行，统计loss时必须读取`batch_features.shape[0]`。

## 5. 完整数据流和shape

一个训练batch经历以下变化：

```text
sample ID列表
→ Dataset逐个读取memmap切片
→ batch_sequences: (B, 14, 200)
→ Conv1d:           (B, 32, 200)
→ ReLU
→ Conv1d:           (B, 32, 200)
→ ReLU
→ masked mean:      (B, 32)
→ Linear:           (B, 1)
→ 预测一个target
```

两层卷积都使用核宽5、步幅1、每侧padding 2。第一层一个位置看5个相邻时刻；第二层继续组合第一层的相邻结果，所以最终感受野是9个时间位置：

$$
5 + (5-1) = 9
$$

这表示一个第二层响应最多综合附近9个记录位置的信息。原始记录并非严格等间隔，所以不能把它绝对解释为固定27秒。

## 6. block shuffle接入DataLoader

普通写法是：

```python
DataLoader(dataset, batch_size=512, shuffle=True)
```

它可能让一个batch里的512个样本来自磁盘各处，对6.56 GiB memmap产生大量随机读取。

现在改成：

```python
train_batch_sampler = BlockShuffleBatchSampler(
    sample_count=len(train_dataset),
    batch_size=batch_size,
    seed=random_seed,
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    num_workers=0,
)
```

`batch_sampler`每次直接给出一整个batch的位置列表，例如`[1024,1025,...,1535]`。因此不能再同时传入`batch_size=`或`shuffle=`，否则DataLoader不知道应该听哪套分批规则。

注意：这里连续的是Dataset位置。我们已经实测训练区间内的sample ID也连续，所以它们最终对应连续的memmap切片。

## 7. 为什么mask-aware pooling仍然需要

短样本采用左padding、真实记录右对齐。卷积层会为所有200个位置计算隐藏响应，但汇总时只允许`row_mask=1`的位置进入平均：

$$
\text{pooled}=
\frac{\sum_t h_t m_t}
{\max(\sum_t m_t,1)}
$$

- $h_t$：某个隐藏通道在位置$t$的响应；
- $m_t$：该位置的`row_mask`，真实行为1，padding为0；
- 分母：真实位置数量；
- `max(...,1)`：防止极端情况下除以0。

block shuffle只改变样本之间的训练顺序，不改变单个`(14,200)`数组内部的时间顺序，因此不会阻止CNN学习局部时间模式。

## 8. 最佳权重的保存与恢复

训练过程中每轮都会得到一个验证余弦分数。只有分数创下新高时才保存：

```python
torch.save(model.state_dict(), best_model_path)
```

`model.state_dict()`返回一个类似字典的对象，里面保存每层参数名及其Tensor，但不保存Python类本身。

训练结束后，新建同结构模型并恢复：

```python
restored_model = MaskAwareCNN().to(device)

saved_state = torch.load(
    best_model_path,
    map_location=device,
    weights_only=True,
)

restored_model.load_state_dict(saved_state)
restored_model.eval()
```

### `torch.load()`

- 来源：PyTorch顶层函数，需要`import torch`。
- 基本调用：`torch.load(path, map_location=..., weights_only=True)`。
- 输入：权重文件路径；`map_location`指定载入设备。
- 返回：本项目中返回参数字典，不是已经能预测的模型。
- 常见错误：以为`torch.load()`会自动知道`MaskAwareCNN`结构。

### `model.load_state_dict()`

- 来源：`nn.Module`对象的方法。
- 基本调用：`model.load_state_dict(saved_state)`。
- 输入：参数名字和shape与当前模型一致的状态字典。
- 返回：载入结果信息；主要作用是原地更新当前模型参数。
- 常见错误：模型层数、通道数或参数名字与保存时不同，导致无法匹配。

恢复后，脚本会重新计算完整验证集预测。恢复分数应与训练时记录的`best cosine`在浮点误差内一致，这证明保存的确实是最佳轮参数。

## 9. 本次脚本如何运行

脚本：`exercises/lesson_06c_full_cnn_train.py`

第一步仍先运行一次短检查：

```powershell
D:\anaconda\envs\pytorch\python.exe exercises\lesson_06c_full_cnn_train.py
```

它使用20,000个训练样本、5,000个验证样本，只跑1轮，检查新接入的block sampler、保存和恢复是否正常。这个分数不能与完整LightGBM比较。

短检查通过后运行完整实验：

```powershell
D:\anaconda\envs\pytorch\python.exe exercises\lesson_06c_full_cnn_train.py --full
```

完整模式使用全部训练/验证样本并训练10轮。运行时每250个训练batch打印一次进度，避免长时间无输出让人误以为卡住。

产物保存在：

```text
data/interim/lesson_06c_full_cnn_runs/<本次运行时间>/
├── best_model.pt
├── best_valid_predictions.feather
├── history.json
└── run_config.json
```

## 10. 本课动手任务

固定训练代码已经提供，不要求你机械重写Dataset、模型和验证循环。你真正要做的是：

1. 先阅读脚本中“block sampler接入DataLoader”的位置，确认训练loader没有同时设置`batch_size`和`shuffle`；
2. 运行短检查，把完整输出发给Codex验收；
3. 短检查通过后运行`--full`；
4. 报告最佳epoch、最佳余弦、每轮时间和产物目录；
5. 根据结果在以下判断中选择合理项，可以多选，并说明证据：

   - A：CNN高于MLP但低于LightGBM，说明序列保留了一些聚合表没有充分利用的信息，但当前结构仍需改进。
   - B：CNN低于LightGBM，不足以证明CNN不适合金融序列；还应检查结构、感受野、输入数据流和优化设置。
   - C：只要训练MSE下降，就可以认定CNN已经超过LightGBM，不必看验证余弦。

其中有两个可能合理判断和一个大概率错误判断。先根据实际结果再决定，不提前追求某个答案。

## 11. 验收标准

短检查必须满足：

- 单样本shape为`(14,200)`，首批shape为`(512,14,200)`；
- sampler检查显示每个训练位置每轮恰好出现一次；
- loss和验证余弦均为有限数；
- 成功保存最佳权重；
- 恢复后的验证余弦与最佳分数一致。

完整实验还必须满足：

- 训练样本数为1,064,163，验证样本数为193,474；
- 完成10轮或明确记录中断原因；
- 使用同一month 0～59 / 60～70划分；
- 报告与MLP `0.044684`、LightGBM `0.103870`的差值；
- 在看到验证证据前，不凭训练loss下结论。
