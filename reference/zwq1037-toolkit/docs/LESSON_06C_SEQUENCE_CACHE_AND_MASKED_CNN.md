# 第 6C 课：把完整序列送进 CNN

## 1. 本课最终目标

第6B课已经能把一个`sample_id`变成固定形状的序列。现在真正的困难是：训练集有1,257,637个样本，若把全部序列一次放入内存，普通电脑很容易撑不住。

第6C课最终要完成：

```text
完整market原始记录
→ 固定长度序列缓存
→ 按月份划分训练/验证
→ mask-aware CNN
→ 训练、验证并与LightGBM比较
```

为了不同时塞入太多新东西，本课分成三个检查点：

- **6C-A（现在）**：理解磁盘缓存与内存映射，在5个真实样本上完成读写；
- **6C-B**：使用固定的资源安全支持代码生成全量缓存；
- **6C-C**：实现mask-aware pooling并训练CNN。

这三个检查点共同属于第6C课，不额外增加重复的CNN入门课。

## 2. 配套阅读

本检查点不安排外部阅读。

原因是本节核心是NumPy磁盘映射接口和当前电脑的资源边界，不是新的神经网络理论。Python/NumPy接口由本讲义完整讲解，不要求你阅读官方API文档。到6C-C建立CNN时，再根据模型结构决定是否补充《动手学深度学习》的对应小节。

读完本讲义后需要回答：

1. 内存映射为什么能让我们使用一个大于可用内存的数组？
2. 内存映射是否会自动压缩文件？
3. 为什么缓存可以保存为`float16`，而送入模型前仍转换成`float32`？

## 3. 遇到了什么麻烦？

我们的序列形状计划为：

```text
(样本数, 通道数, 时间位置数)
=(1,257,637, 14, 200)
```

如果使用`float32`，每个数占4字节：

$$
1{,}257{,}637\times14\times200\times4
\approx 13.1\ \text{GiB}
$$

这还只是特征本身。Python、PyTorch、标签、模型和操作系统也需要内存，因此不能把它作为普通NumPy数组一次常驻在16GB内存中。

### 人话解释

普通数组像是把整本词典的每一页都摊在书桌上。词典虽然能放进房间，但书桌放不下。

磁盘映射数组（memory-mapped array）像是词典仍放在书架上：程序保留页码，需要哪几页时才把那些页调进内存。对代码来说它仍然很像数组，可以使用`cache[100:132]`取一批样本。

关键边界：

- 它减少的是**同时占用的内存**；
- 它不会让全部数据消失，也不会自动压缩磁盘文件；
- 第一次访问某一部分时仍需要从磁盘读取，所以磁盘速度会影响训练。

## 4. 为什么缓存用`float16`？

`float16`每个数占2字节，理论磁盘大小约为：

$$
1{,}257{,}637\times14\times200\times2
\approx 6.56\ \text{GiB}
$$

这次缓存里的连续特征已经过标准化，`row_mask`又只有0和1，因此第一版选择`float16`减少一半磁盘和读取量。

但训练时我们会把一个batch转换回`float32`：

```python
batch_features = batch_features.to(dtype=torch.float32, device=device)
```

原因是：

- `float16`缓存解决的是存储和读取压力；
- `float32`训练提供更稳妥的数值范围和精度；
- 一次只转换一个batch，不会重新把6.56 GiB全部复制进内存。

这里存在轻微量化误差，所以正式实验必须记录缓存精度。若结果可疑，可以再用`float32`缓存作对照，不能把所有分数变化都归因于CNN。

## 5. 新工具一：`np.dtype(...).itemsize`

### 解决什么问题

查询一种NumPy数据类型的每个元素占多少字节，用于训练前估计资源。

### 来源与导入

来自NumPy：

```python
import numpy as np
```

### 基本语法

```python
bytes_per_value = np.dtype(np.float32).itemsize
```

- 输入：NumPy数据类型，如`np.float32`或`np.float16`；
- 返回：Python整数；
- `np.float32`返回4，`np.float16`返回2。

### 最小例子

```python
shape = (10, 3)
number_of_values = 10 * 3
number_of_bytes = number_of_values * np.dtype(np.float32).itemsize
print(number_of_bytes)  # 120
```

### 当前比赛中的作用

在真正创建缓存前，先根据`shape × 每个值的字节数`估算约需多少磁盘空间。

### 常见错误

`itemsize`是属性，不是方法，后面没有括号：

```python
np.dtype(np.float32).itemsize      # 正确
np.dtype(np.float32).itemsize()    # 错误
```

## 6. 新工具二：`np.lib.format.open_memmap()`

### 解决什么问题

创建或打开一个磁盘上的`.npy`数组，让我们像操作NumPy数组一样分块写入，而不必先在内存中造出完整大数组。

### 来源与导入

来自NumPy，不需要额外安装：

```python
import numpy as np
```

### 基本调用

```python
cache = np.lib.format.open_memmap(
    cache_path,
    mode="w+",
    dtype=np.float16,
    shape=(5, 14, 200),
)
```

主要参数：

- `cache_path`：`Path`或字符串，输出文件位置；
- `mode="w+"`：创建一个可读写文件；如果文件已存在，会覆盖它；
- `dtype=np.float16`：每个元素的数据类型；
- `shape=(5,14,200)`：完整数组形状。

返回值是`numpy.memmap`对象，shape仍然是`(5,14,200)`。它支持普通切片赋值：

```python
cache[0:3] = source_data[0:3]
cache[3:5] = source_data[3:5]
```

这两次只写入各自的切片。未来全量构建也是同一思路：一次准备一块，写完便复用内存。

### 最小例子

```python
small_cache = np.lib.format.open_memmap(
    "small.npy",
    mode="w+",
    dtype=np.float32,
    shape=(3, 2),
)
small_cache[0] = [1.0, 2.0]
small_cache[1:3] = [[3.0, 4.0], [5.0, 6.0]]
small_cache.flush()
```

### 当前比赛中的作用

把完整的`(N,14,200)`序列逐块写到磁盘，避免13.1 GiB普通数组常驻内存。

### 最常见的错误

`mode="w+"`会覆盖同名缓存。在正式构建器中必须先检查路径与配置，不能无意中覆盖一份耗时很久生成的正确缓存。

## 7. 新方法：`cache.flush()`

### 解决什么问题

要求操作系统把尚在缓冲区中的修改写向磁盘。

### 调用语法与返回值

```python
cache.flush()
```

它是`numpy.memmap`对象的方法，不需要参数，返回`None`，不会生成一个新数组。

### 最小例子与比赛用途

写完一批或完成整个缓存后调用：

```python
cache[0:100] = one_block
cache.flush()
```

常见误解：`flush()`不等于关闭文件，也不等于删除变量。它只是把修改同步到磁盘。

## 8. 新工具三：`np.load(..., mmap_mode="r")`

### 解决什么问题

以只读内存映射方式重新打开已经保存的`.npy`缓存。

### 基本调用

```python
reopened_cache = np.load(cache_path, mmap_mode="r")
```

- `cache_path`：`.npy`文件路径；
- `mmap_mode="r"`：read-only，只读；
- 返回：`numpy.memmap`对象，保留原来的shape和dtype。

### 最小例子

```python
reopened = np.load("small.npy", mmap_mode="r")
first_row = reopened[0]
print(first_row)
```

### 当前比赛中的作用

训练脚本只读缓存，避免训练过程误改数据。DataLoader以后会根据样本编号读取其中的一小批。

### 常见错误

省略`mmap_mode`：

```python
ordinary_array = np.load(cache_path)
```

通常会把整个数组作为普通数组载入内存。对5个演示样本没关系，对数GiB正式缓存就可能造成巨大内存压力。

## 9. 为什么缓存直接保存成`(N,C,T)`？

第6B先得到`(N,T,C)`，然后使用`permute(0,2,1)`变成Conv1d需要的`(N,C,T)`。

正式缓存将直接保存最终训练布局：

```text
(样本, 通道, 时间) = (N, 14, 200)
```

这样每次读取batch后不必再交换轴。三个轴分别表示：

- 轴0：第几个样本；
- 轴1：14个输入通道；
- 轴2：200个时间位置。

## 10. 本检查点的最小示例流程

```text
第6B产生的5个真实序列 (5,14,200)
→ 建立float16 .npy磁盘映射
→ 分两块写入
→ flush
→ 删除当前Python引用
→ 以只读mmap重新打开
→ 检查shape、dtype、mask和数值误差
```

`float32 → float16`会发生舍入，所以连续特征不能要求逐位完全相同。我们用最大绝对误差检查它没有发生明显破坏；0/1的`row_mask`则应当精确保持。

## 11. 你的动手任务

打开：

```text
exercises/lesson_06c_sequence_cache.py
```

只完成四个TODO：

1. 计算完整`float16`缓存的字节数；
2. 用`open_memmap()`建立5样本演示缓存；
3. 把5个真实序列分成两块写入；
4. 用`np.load(..., mmap_mode="r")`只读重开。

这里亲手写分块赋值有价值，因为全量缓存只是把“两块”扩展成“很多块”。演示数据准备、数值检查和输出已提供，不要求机械重写第6B序列转换逻辑。

### 预期输出

```text
full float16 cache GiB = 6.56 左右
source shape = (5, 14, 200)
cache shape = (5, 14, 200)
cache dtype = float16
cache object type = <class 'numpy.memmap'>
row mask is exact = True
maximum absolute conversion error = 一个很小的有限数
```

### 验收标准

- 文件能够运行到底，所有断言通过；
- 缓存对象类型是`numpy.memmap`，不是一次载入的普通`ndarray`；
- shape和dtype正确；
- 能解释内存映射减少的是内存峰值，不是磁盘大小；
- 能解释`w+`的覆盖风险与`r`的只读含义。

完成后只需告诉我“已完成6C-A”。我会先读取并运行你的文件，不会直接替你修改。

---

# 检查点 B：从试跑到完整缓存

## 12. 本检查点目标

你已经用5个样本亲手完成了磁盘映射。现在不再机械重写数据工程代码，而是学会安全启动、检查和恢复一个规模化构建任务。

固定脚本：

```text
scripts/build_train_market_sequence_cache.py
```

它会完成：

```text
读取每个样本的行数
→ 审计原始sample_id排列
→ 建立(N,14,200)缓存
→ 每个子进程只读取一个原始列
→ 仅用训练月份真实行计算mean/std
→ 分块写入一个通道
→ 子进程退出并释放整列内存
→ 在manifest.json记录完成状态
```

## 13. 为什么固定代码一次只读取一个原始列？

原始market有2.217亿行。若同时读取13列，内存压力很大。脚本改成：

```text
子进程1：只读transaction_avgprice → 写通道0 → 退出
子进程2：只读transaction_volume   → 写通道1 → 退出
……
```

每个子进程退出后，操作系统会回收它读取的完整列。这样峰值内存由“所有列加在一起”变成“一个原始列 + 一个小输出块”。

这和上一检查点的分块并不重复：

- **按列隔离**控制解压原始宽表的内存；
- **按样本分块写入**控制构造`(样本,200)`输出的内存。

## 14. 什么是smoke test？

smoke test（冒烟测试）不是正式实验。它只用前1000个真实样本，目的像“正式点火前先看机器会不会冒出异常的烟”。

它检查：

- 路径能否读取；
- 原始行是否确实按连续`sample_id`分组；
- 每组倒计时是否从大到小；
- 14个通道能否写完；
- 缓存是否可只读重开；
- 中断状态是否能记录。

试跑分数没有意义，而且试跑统计量只来自这1000个样本，不能拿去训练正式模型。

## 15. `manifest.json`是做什么的？

缓存文件有6.56 GiB，构建过程中可能因为关机、内存不足或程序异常而中断。如果只有一个`sequences.npy`，我们无法知道哪些通道已经完整写好。

`manifest.json`是缓存的施工记录，例如：

```json
{
  "status": "building",
  "shape": [1000, 14, 200],
  "dtype": "float16",
  "completed_channels": ["row_mask", "transaction_avgprice"]
}
```

再次运行时，脚本会跳过已经登记完成的通道，从未完成处继续。只有全部通道成功后，`status`才会变为`complete`。

重要边界：manifest不是证明数值一定正确的魔法。脚本只有在子进程成功返回后才登记通道完成，并且还会检查布局、shape、dtype与有限数值。

## 16. 新工具：独立子进程

主脚本内部使用：

```python
subprocess.run(command, check=True)
```

这一段由固定代码提供，本课不要求你亲手重写。

- `subprocess`来自Python标准库；
- `command`是字符串列表，描述要启动的Python命令；
- `check=True`表示子进程非正常退出时立刻抛出错误，主脚本不能把失败通道登记为完成；
- 返回值是`CompletedProcess`对象，本脚本不需要使用它；
- 子进程有自己的内存空间，退出后比只写`del`更可靠地归还大列内存。

最小例子：

```python
import subprocess
import sys

command = [sys.executable, "small_job.py"]
subprocess.run(command, check=True)
```

常见错误：写成`check=False`后忽略失败，后续流程可能继续使用不完整输出。

## 17. 训练统计量边界

正式缓存包括月份0～70，因为训练与验证都要读取序列；但是均值和标准差只能使用月份0～59对应的真实market行：

```text
月份0～59：fit mean/std，再执行transform
月份60～70：只用已有mean/std执行transform
padding：不参与统计，保持0
```

验证行虽然会被写进同一个缓存文件，但“存放在一起”不等于“共同拟合统计量”。数据泄露取决于计算过程是否使用了验证信息，不取决于是否处于同一个文件。

## 18. 本检查点任务：先运行试跑

在PyCharm中直接运行：

```text
scripts/build_train_market_sequence_cache.py
```

不要添加`--full`。默认就是1000样本试跑，输出目录与正式缓存完全分开：

```text
data/interim/sequence_cache_smoke_v1/
```

运行结束应看到：

```text
layout audit passed
countdown order audit passed
cache status = complete
cache shape = (1000, 14, 200)
cache dtype = float16
SMOKE ONLY
```

第一次运行会完成全部通道。随后原样再运行一次，应该看到多条`reusing completed source column=...`，这证明断点续建状态被读取，而不是重新处理所有列。

## 19. 你需要提交的观察

不用修改固定构建脚本。运行两次后告诉我：

1. 第一次运行最后的`cache status`、shape、dtype和三个抽查样本的observed counts；
2. 第二次是否出现`reusing completed source column`；
3. 为什么月份60～70可以存进同一缓存，却不能参与mean/std计算？

验收后，我会检查缓存文件和manifest；确认无误，再由你明确启动`--full`。完整构建可能较久，所以不会在smoke test未通过前创建6.56 GiB正式文件。

---

# 检查点 C1：让平均池化真正忽略 padding

## 20. 本检查点目标

正式缓存已经完成，但还不能直接把第6A的`AdaptiveAvgPool1d(1)`原样搬过来。本检查点只解决一个核心问题：

> CNN得到200个位置的隐藏响应以后，怎样只平均真实位置，不让左侧padding参与结果？

完成后你将能够：

1. 从`(B,14,200)`输入中取出形状仍为`(B,1,200)`的`row_mask`；
2. 实现mask-aware mean pooling；
3. 用数字例子证明修改padding位置不会改变池化结果；
4. 在真实缓存的一个小batch上完成CNN前向传播。

本检查点通过后，C2直接把它接入正式磁盘Dataset和训练循环。

## 21. 配套阅读

本检查点不增加外部阅读。

《动手学深度学习》的普通池化章节主要讲固定窗口的最大值/平均值，并不直接解决当前这种“不同样本有不同数量padding”的masked pooling。你已经学过`AdaptiveAvgPool1d(1)`的普通平均，本讲义针对当前数据完整讲解差异，避免为了形式重复阅读。

读完后回答：

1. 为什么把`row_mask`送进CNN，不等于CNN会自动忽略padding？
2. 为什么取mask时使用`13:14`，而不是只写`13`？
3. masked mean的分母为什么是有效位置数，而不是固定的200？

## 22. 先看一个会出错的普通平均

假设某个卷积通道产生4个隐藏响应：

```text
hidden = [100, 100, 2, 4]
mask   = [  0,   0, 1, 1]
```

前两个位置是padding，后两个位置是真实记录。

普通平均会计算：

$$
\frac{100+100+2+4}{4}=51.5
$$

但我们真正想要的是只平均真实位置：

$$
\frac{2+4}{2}=3
$$

这里用100是为了让错误明显。真实CNN的padding位置即使输入全0，卷积偏置和邻近真实位置也可能让其隐藏响应不为0，所以不能假设padding的隐藏响应天然为0。

## 23. masked mean pooling公式

对某个样本、某个隐藏通道，masked mean为：

$$
\text{pooled}=\frac{\sum_{t=1}^{T}h_t m_t}{\sum_{t=1}^{T}m_t}
$$

每个符号的含义：

- $T=200$：序列总位置数；
- $h_t$：CNN在第$t$个位置产生的隐藏响应；
- $m_t$：该位置的`row_mask`，真实位置为1，padding为0；
- $h_tm_t$：padding响应乘0后被排除；
- $\sum m_t$：真实位置数量，例如107，而不是固定200。

仍用刚才的数字：

$$
\frac{100\times0+100\times0+2\times1+4\times1}{0+0+1+1}=3
$$

## 24. 为什么切片写成`13:14`？

输入shape是：

```text
(batch, channels, time) = (B, 14, 200)
```

两种写法的结果不同：

```python
mask_without_channel_axis = batch_sequences[:, 13, :]
# shape: (B, 200)

row_mask = batch_sequences[:, 13:14, :]
# shape: (B, 1, 200)
```

Python切片`13:14`包含索引13，但不包含14。因为使用的是切片而不是单个整数索引，PyTorch会保留这一轴，所以通道轴仍然存在，长度为1。

我们需要`(B,1,T)`，因为它要同时作用于每个隐藏通道。

## 25. 广播：一个mask怎样控制32个隐藏通道？

假设：

```text
hidden_features.shape = (B, 32, 200)
row_mask.shape         = (B,  1, 200)
```

执行：

```python
masked_features = hidden_features * row_mask
```

PyTorch看到中间一轴分别为32和1，会把长度为1的mask通道**广播（broadcast）**到32个隐藏通道：同一条时间mask同时乘到该样本的32条响应序列。

结果shape仍为：

```text
(B, 32, 200)
```

广播不是实际复制32份大型Tensor，而是按照兼容shape执行逐元素运算。

最小例子：

```python
features = torch.tensor([
    [[10.0, 20.0],
     [30.0, 40.0]],
])
# shape = (1, 2, 2)

mask = torch.tensor([[[0.0, 1.0]]])
# shape = (1, 1, 2)

print(features * mask)
```

结果：

```text
[[[ 0, 20],
  [ 0, 40]]]
```

同一个mask同时控制了两个特征通道。

常见错误：如果两个shape无法从最后一轴向前匹配，PyTorch会报尺寸不兼容错误。保留mask的通道轴能让含义和shape更清楚。

## 26. 新方法：`tensor.sum(dim=...)`

### 解决什么问题

沿指定轴求和。

### 来源

它是PyTorch Tensor的方法，已有Tensor即可调用：

```python
import torch
```

### 基本语法

```python
summed_features = masked_features.sum(dim=2)
```

- 输入对象：`masked_features`，shape为`(B,H,T)`；
- `dim=2`：沿第2轴，也就是时间轴求和；
- 返回：新Tensor，shape为`(B,H)`；
- 原Tensor不会被修改。

例如`(8,32,200)`沿时间轴求和后得到`(8,32)`，每个样本的每个隐藏通道剩下一个总和。

常见错误：写成`dim=1`会把隐藏通道加在一起，得到`(B,T)`，含义完全不同。Python轴从0开始编号。

## 27. 新方法：`tensor.clamp_min()`

### 解决什么问题

把小于某个下限的值提升到该下限，用于防止分母为0。

### 基本语法

```python
valid_counts = row_mask.sum(dim=2).clamp_min(1.0)
```

- `row_mask.sum(dim=2)`得到`(B,1)`的有效位置数；
- `.clamp_min(1.0)`返回同shape的新Tensor；
- 原来大于等于1的计数不变，0会被替换成1。

最小例子：

```python
values = torch.tensor([0.0, 2.0, 5.0])
print(values.clamp_min(1.0))
# tensor([1., 2., 5.])
```

当前缓存保证每个样本至少有一条真实记录，所以正常情况下分母不会为0。这里仍加保护，是为了让函数面对异常输入时不产生除零和`NaN`。

常见误解：它不会把所有计数都变成1；只修改低于1的值。

## 28. 完整shape链条

```text
batch_sequences                         (B, 14, 200)
│
├─ [:, 13:14, :] → row_mask             (B,  1, 200)
│
└─ Conv1d → hidden_features             (B, 32, 200)
                  │
                  × row_mask（广播）     (B, 32, 200)
                  │
                  sum(dim=2)             (B, 32)
                  ÷ mask.sum(dim=2)      (B, 32)
                  │
                  Linear(32,1)           (B,  1)
```

这里最终仍然是一条连续回归预测，不是分类概率。

## 29. 模型类由谁写？

练习已经提供`MaskAwareCNN(nn.Module)`骨架和卷积层。你之前已经理解类、`self`、`forward()`与`model(batch)`，本检查点不要求重复搭建全部模型。

你只亲手完成四个紧密相关的TODO：

1. 用`13:14`取出mask并保留通道轴；
2. 隐藏响应乘mask后沿时间轴求和；
3. 对mask沿时间轴计数并用`clamp_min(1.0)`保护；
4. 总和除以有效计数。

## 30. 练习与验收

打开：

```text
exercises/lesson_06c_mask_aware_pooling.py
```

运行成功后应看到类似：

```text
ordinary pooled = [[51.5, 31.0]]
masked pooled = [[3.0, 12.0]]
padding invariance = True
real batch shape = (4, 14, 200)
real output shape = (4, 1)
```

`padding invariance=True`的含义是：只修改mask为0的位置，池化结果完全不变。这正是本课核心验收，而不是只看代码没有报错。

完成后回答三道复盘题并告诉我“已完成6C-C1”。我会先运行你的文件，不直接修改；通过后下一步直接接正式缓存Dataset与训练。

---

# 检查点 C2：从磁盘按需读取，并启动 CNN 训练

## 31. 本检查点目标

现在三块已经齐了：

```text
正式序列缓存       (1,257,637, 14, 200)
mask-aware CNN     (B, 14, 200) → (B, 1)
已有训练验证流程    MSE训练、cosine选最佳轮
```

最后缺少的是中间的“取数据”部分：不能把6.56 GiB缓存复制成一个普通float32 Tensor，因为那会扩大到约13.1 GiB。我们要让DataLoader每次只取得一个batch。

本检查点完成后，你会：

1. 理解自定义`Dataset`的`__len__()`与`__getitem__()`；
2. 从只读memmap按`sample_id`取得单个`(14,200)`序列；
3. 只把当前样本转换成独立的float32 Tensor；
4. 运行20,000/5,000真实时间划分的3轮CNN smoke训练。

## 32. 配套阅读

本检查点不增加外部阅读。

你已经在第5B课掌握`TensorDataset`、`DataLoader`、batch和shuffle。这次新增的是当前项目特有的磁盘按需读取边界；Python/PyTorch接口由本讲义完整讲解，不要求阅读官方文档。训练循环、MSE、Adam、验证模式、`no_grad`、cosine和最佳权重保存均已验收，不重复安排教材章节。

读完后回答：

1. 为什么Dataset返回前要复制当前样本，而不是修改只读memmap？
2. `position`和`sample_id`为什么不一定是同一个数字？
3. 为什么smoke验证分数不能直接和完整LightGBM分数比较？

## 33. DataLoader到底怎样向Dataset要数据？

以前使用：

```python
dataset = TensorDataset(features, targets)
```

这里`features`和`targets`已经是准备好的Tensor。DataLoader本质上反复做类似的事情：

```python
dataset[第几个位置]
```

自定义Dataset就是由我们规定：收到一个位置以后，去哪里找到特征和标签，并返回什么。

简化流程：

```text
DataLoader决定这批要位置[7, 2, 9]
        ↓
dataset.__getitem__(7)
dataset.__getitem__(2)
dataset.__getitem__(9)
        ↓
每次从memmap复制一个(14,200)样本
        ↓
DataLoader拼成(3,14,200) batch
```

DataLoader仍然负责batch和shuffle；Dataset只负责回答“第这个位置的数据是什么”。

## 34. 新类：`torch.utils.data.Dataset`

### 解决什么问题

为DataLoader提供统一的数据访问规则，特别适合特征不能提前全部变成内存Tensor的情况。

### 来源与导入

```python
from torch.utils.data import Dataset
```

它来自PyTorch。我们通过继承它建立自己的类：

```python
class SmallDataset(Dataset):
    ...
```

### 必须提供的两个特殊方法

```python
def __len__(self):
    return 样本数量

def __getitem__(self, position):
    return 一个样本的特征, 一个样本的标签
```

双下划线方法由Python或框架自动调用：

- `len(dataset)`会调用`dataset.__len__()`；
- `dataset[5]`会调用`dataset.__getitem__(5)`；
- 通常不要在外面手写`dataset.__getitem__(5)`。

### 最小例子

```python
from torch.utils.data import Dataset


class NumberDataset(Dataset):
    def __init__(self):
        self.features = [10, 20, 30]
        self.targets = [1, 0, 1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, position):
        return self.features[position], self.targets[position]
```

使用：

```python
dataset = NumberDataset()
print(len(dataset))   # 3
print(dataset[1])     # (20, 0)
```

### 当前比赛的输入与返回

我们的`MemmapSequenceDataset`保存：

- `sequence_cache`：shape `(1_257_637,14,200)`的只读`numpy.memmap`；
- `sample_indices`：这个Dataset允许访问哪些真实sample ID；
- `all_targets`：按sample ID排列的一维target数组。

每次`__getitem__()`返回：

- 特征Tensor：`(14,200)`、`torch.float32`；
- 标签Tensor：`(1,)`、`torch.float32`。

DataLoader拼接后得到：

- batch特征：`(B,14,200)`；
- batch标签：`(B,1)`。

### 常见错误

`__len__()`必须返回Python整数；`__getitem__()`必须返回一个样本，不能每次返回完整缓存。

## 35. `position`不一定等于`sample_id`

假设验证Dataset只包含：

```python
sample_indices = [1064163, 1064164, 1064165]
```

调用：

```python
dataset[0]
```

这里：

```text
position  = 0
sample_id = 1064163
```

`position`表示“在当前Dataset中排第几个”；`sample_id`表示“在完整缓存的轴0中取哪一行”。所以需要先映射：

```python
sample_id = int(self.sample_indices[position])
```

如果直接写`self.sequence_cache[position]`，验证Dataset的第0项会错误读取训练样本ID 0。

## 36. 新工具：`np.array(..., dtype=..., copy=True)`

### 解决什么问题

把memmap中的一个只读float16样本复制为独立的float32普通数组。

### 来源和语法

来自NumPy：

```python
import numpy as np

feature_array = np.array(
    self.sequence_cache[sample_id],
    dtype=np.float32,
    copy=True,
)
```

主要输入：

- 第一个参数：一个`(14,200)`的memmap切片；
- `dtype=np.float32`：转换后的元素类型；
- `copy=True`：建立独立内存，不与只读缓存共享存储。

返回：普通`numpy.ndarray`，shape仍为`(14,200)`，dtype为float32。

### 最小例子

```python
small_half = np.array([1.5, 2.5], dtype=np.float16)
small_float = np.array(small_half, dtype=np.float32, copy=True)
```

### 当前比赛中的作用

缓存保留float16以减少磁盘读取量；只有当前访问的单个样本变成float32，随后`torch.from_numpy()`与该小数组共享内存。

### 常见错误

不要写：

```python
all_float32 = np.array(self.sequence_cache, dtype=np.float32, copy=True)
```

这会尝试一次复制完整缓存，正是我们要避免的约13.1 GiB内存占用。

## 37. 为什么本课先保持`num_workers=0`？

Windows的DataLoader多进程会为worker启动新的Python进程。自定义memmap Dataset虽然可以进一步设计成每个worker安全重开文件，但那会同时引入进程序列化、worker生命周期和并发磁盘读取。

本课先使用：

```python
num_workers=0
```

含义是由当前训练进程调用Dataset，先验证正确性。性能是否需要增加worker，要根据正式训练的GPU等待情况再做单变量实验，不能先把复杂度堆上去。

## 38. smoke训练如何划分？

先遵守完整时间边界：

```text
训练候选：month 0～59
验证候选：month 60～70
```

然后分别在各自范围内取一小段连续sample ID：

```text
smoke train：训练范围开头连续20,000个
smoke valid：验证范围开头连续 5,000个
```

这样不会把验证月份混入训练，而且能让机械硬盘/SSD顺序读取相邻序列，避免第一次管线检查被随机寻址拖慢。代价是这两段数据不代表完整分布，因此smoke cosine更不能与使用全部数据训练的LightGBM `0.103870`公平比较。

本次smoke的训练DataLoader也暂时设置`shuffle=False`。这不是认定正式训练永远不需要打乱，而是先证明Dataset、模型和验证闭环正确。正式训练前会使用“打乱batch块、块内顺序读取”的策略，在优化随机性和磁盘吞吐之间取得平衡。

## 39. 本课复用而不重写的部分

练习已经提供：

- 已验收的masked mean和CNN；
- 时间划分与固定抽样；
- DataLoader；
- MSE、Adam及3轮训练；
- `eval()`、`no_grad()`和全体验证cosine；
- 最佳权重及日志保存。

这些都不是本次新能力，不要求你机械重写。

## 40. 你的四个TODO

打开：

```text
exercises/lesson_06c_cnn_memmap_train.py
```

完成`MemmapSequenceDataset`中的四处：

1. `__len__()`返回当前Dataset的位置数量；
2. 把Dataset位置映射为真实`sample_id`；
3. 只复制这个sample ID的序列并转成float32；
4. 从`all_targets`中取对应标签并建立shape `(1,)`的float32 Tensor。

运行前保持：

```python
SMOKE_TEST = True
```

验收关注：

- 单样本shape为`(14,200)`和`(1,)`；
- batch shape正确；
-3轮loss和cosine均为有限数；
-最佳权重成功保存；
-能解释Dataset位置与sample ID的区别。

本课运行产物保存到`data/interim/lesson_06c_cnn_runs/`。项目旧`outputs`目录存在独立权限边界，这只是输出位置选择，不改变训练数据、模型或评分规则。

完成后告诉我“已完成6C-C2 smoke”。我会读取代码、运行并检查产物，不直接修改。smoke通过后再决定正式epoch、batch size和读取策略。

## 40.1 用同一个三样本例子贯穿四个TODO

前面的讲解把函数分别介绍了，但真正写TODO时还需要看清三份成员数据怎样连接。先完全离开比赛大数据，假设对象内部是：

```python
self.sample_indices = np.array([10, 25, 80])
self.all_targets = np.array([0.0, 0.1, 0.2, ..., 8.0, ...])
self.sequence_cache.shape = (100, 14, 200)
```

这里：

- 完整缓存有100个样本，轴0编号为`0～99`；
- 完整target也有100个，按同一个sample ID编号；
- 当前Dataset只选择了sample ID `10、25、80`，所以它只有3个位置。

对应关系是：

| Dataset position | `self.sample_indices[position]` | 实际读取缓存 | 实际读取标签 |
|---:|---:|---|---|
| 0 | 10 | `sequence_cache[10]` | `all_targets[10]` |
| 1 | 25 | `sequence_cache[25]` | `all_targets[25]` |
| 2 | 80 | `sequence_cache[80]` | `all_targets[80]` |

### TODO 1：Dataset长度看哪一个对象？

当前Dataset只有`sample_indices`中列出的3个样本。因此：

```text
len(self.sample_indices) = 3        正确
len(self.all_targets) = 100         错误：这是完整数据的标签数量
len(self.sequence_cache) = 100      错误：这是完整缓存的样本数量
```

如果错误返回100，DataLoader迟早会要求`dataset[50]`；但`sample_indices`只有位置0、1、2，访问`sample_indices[50]`就会越界。

回到smoke：完整target有1,257,637个，但训练Dataset只选择20,000个，所以`len(train_dataset)`必须是20,000。

### TODO 2：把position翻译成sample ID

假设DataLoader调用：

```python
dataset[1]
```

传入的是`position=1`。先查询映射表：

```python
self.sample_indices[position]
```

得到NumPy整数`25`。可以再用Python内置函数`int()`把它转换成普通Python整数：

```python
sample_id = int(self.sample_indices[position])
```

`int(value)`不需要import，输入一个可转换成整数的数，返回Python `int`。这里转换不是改变sample ID，而是让后续索引类型更明确。常见错误是直接写`sample_id = position`，那会把Dataset位置1错当成完整缓存的sample ID 1。

### TODO 3：只复制这个sample ID的序列

现在已经知道`sample_id=25`。先从完整缓存取这一行：

```python
selected_sequence = self.sequence_cache[sample_id]
```

其shape为`(14,200)`，仍来自只读float16 memmap。然后只转换这一小块：

```python
feature_array = np.array(
    selected_sequence,
    dtype=np.float32,
    copy=True,
)
```

返回的是独立的float32 `numpy.ndarray`，shape仍为`(14,200)`。固定下一行再将它变成Tensor：

```python
feature_tensor = torch.from_numpy(feature_array)
```

不要对`self.sequence_cache`整体调用`np.array`；那会复制全部125万个样本。

### TODO 4：用同一个sample ID取得答案

特征读的是sample ID 25，因此答案也必须读25：

```python
target_value = self.all_targets[sample_id]
```

`target_value`只是一个标量。如果直接写：

```python
torch.tensor(target_value)
```

会得到0维Tensor，shape为`()`。为了让每个样本标签shape为`(1,)`，把标量放入一个单元素列表：

```python
target_tensor = torch.tensor(
    [target_value],
    dtype=torch.float32,
)
```

DataLoader把512个这样的`(1,)`标签叠起来，就得到batch标签`(512,1)`，与模型预测shape相同。

### 四步连起来的人话版本

```text
TODO 1：我这份名单一共有几个人？
TODO 2：名单第position个人的真实sample ID是多少？
TODO 3：按这个sample ID去完整特征仓库拿序列。
TODO 4：按同一个sample ID去完整答案仓库拿标签。
```

真正必须守住的不是某一行语法，而是TODO 3和TODO 4必须使用**同一个sample ID**，否则模型会拿甲的行情去学习乙的答案。

---

# 检查点 C3：打乱batch块，而不是随机跳读每个样本

## 41. 本检查点目标

C2已经证明CNN训练闭环可用。正式训练有约106万个训练样本，现在需要同时满足两个要求：

1. 每个epoch不要永远用完全相同的batch顺序；
2. 不要让磁盘在125万个位置之间逐样本随机跳转。

解决方法是block shuffle（块级打乱）：

```text
先把Dataset位置切成连续batch块
→ 只打乱这些块的顺序
→ 每个块内部仍按连续位置读取
```

## 42. 配套阅读

本检查点不增加外部阅读。

你已经掌握DataLoader的普通shuffle和Python迭代器。本节是当前6.56GiB memmap的I/O策略，不是新的CNN理论；由讲义完整解释`batch_sampler`、`torch.randperm()`与`yield`在这里的具体作用。

读完后回答：

1. block shuffle随机改变的是什么，保持连续的又是什么？
2. 为什么它不改变训练集与验证集的月份边界？
3. 为什么最后一个batch可能不足512个样本？

## 43. 三种读取顺序有什么区别？

假设训练Dataset有10个位置，batch size为4。

### 完全顺序

```text
[0,1,2,3] → [4,5,6,7] → [8,9]
```

磁盘读取连续，但每个epoch顺序完全相同。

### 逐样本随机

```text
[7,0,9,2] → [5,1,8,4] → [3,6]
```

随机性强，但一个batch内部就在文件不同位置跳来跳去。缓存大于可用空闲内存时，随机寻址可能让GPU等待数据。

### block shuffle

先建立三个连续块：

```text
块0=[0,1,2,3]
块1=[4,5,6,7]
块2=[8,9]
```

只打乱块编号，例如得到`[2,0,1]`：

```text
[8,9] → [0,1,2,3] → [4,5,6,7]
```

块的先后顺序变了，但每一块内部仍连续。

## 44. `batch_sampler`是什么？

普通DataLoader写法：

```python
DataLoader(dataset, batch_size=512, shuffle=True)
```

DataLoader自己选择单个位置，再把它们拼成batch。

使用`batch_sampler`时，我们直接把“一整个batch的位置列表”交给DataLoader：

```python
DataLoader(dataset, batch_sampler=block_batch_sampler)
```

如果sampler产生：

```python
[8, 9]
```

DataLoader就调用`dataset[8]`和`dataset[9]`，再拼成一个batch。

重要规则：指定`batch_sampler`后，不能再同时给DataLoader设置`batch_size`、`shuffle`或`sampler`，因为batch如何组成已经完全由`batch_sampler`决定。

## 45. `yield`在人话里是什么？

普通`return`会返回一个结果并结束函数。`yield`会交出一个结果，但保留函数进度，下次继续从后面执行。

最小例子：

```python
def give_numbers():
    yield 10
    yield 20

number_iterator = give_numbers()
print(next(number_iterator))  # 10
print(next(number_iterator))  # 20
```

`give_numbers()`返回的是迭代器。当前sampler每次`yield`一个位置列表，DataLoader便得到一个batch。

## 46. 新函数：`torch.randperm()`

### 解决什么问题

生成从0开始、没有重复的随机排列。

### 来源与导入

来自PyTorch：

```python
import torch
```

### 基本语法

```python
order = torch.randperm(3)
```

输入`3`是元素数量，返回一维整数Tensor，shape为`(3,)`。可能结果：

```text
tensor([2, 0, 1])
```

每个数字`0、1、2`恰好出现一次。

本课使用固定generator：

```python
generator = torch.Generator()
generator.manual_seed(seed + epoch_number)
order = torch.randperm(block_count, generator=generator)
```

同一个seed可以复现实验；加上epoch编号，使不同epoch得到不同顺序。

常见错误：`torch.randint()`可能重复或遗漏编号，不适合保证每个块恰好访问一次。

## 47. 一个块的开始与结束

对于：

```text
sample_count=10
batch_size=4
```

块编号与位置为：

| block_number | start | end | `range(start,end)` |
|---:|---:|---:|---|
| 0 | 0 | 4 | 0,1,2,3 |
| 1 | 4 | 8 | 4,5,6,7 |
| 2 | 8 | 10 | 8,9 |

公式是：

```python
block_start = block_number * batch_size
block_end = min(block_start + batch_size, sample_count)
```

`min(a,b)`是Python内置函数，不需要import，返回两个输入中较小的值。最后一个块若理论结束为12，就用`min(12,10)`截在10，避免越界。

`range(block_start, block_end)`同样不包含右端点。`list(...)`把range转换成DataLoader需要的位置列表。

## 48. 为什么这不是时间泄露？

我们先完成了时间划分：

```text
训练Dataset：month 0～59
验证Dataset：month 60～70
```

block shuffle只改变训练Dataset内部样本被访问的先后顺序，没有把任何验证sample ID加入训练名单。因此它不会破坏月份边界。

“金融数据不能随机划分”不等于“训练集内部永远不能改变batch顺序”：

- 不能随机划分，是为了防止未来样本进入训练、过去样本进入验证；
- 训练内部打乱，是优化器如何遍历已经合法选出的训练样本。

## 49. 你的任务

打开：

```text
exercises/lesson_06c_block_shuffle.py
```

固定代码已经写好随机块编号、两轮迭代与完整性断言。你只完成构造一个连续块的三个TODO：

1. 根据块编号计算`block_start`；
2. 计算不超过`sample_count`的`block_end`；
3. 建立连续位置列表并`yield`给DataLoader。

变量在代码旁标明类型、数值例子和来源。预期每轮产生3个batch，大小为4、4、2；两轮块顺序不同，但每轮把0～9各访问一次，块内相邻差始终为1。

完成后告诉我“已完成C3”。验收通过后，固定支持代码会把同一个sampler接到正式CNN训练，不再增加准备课程。
