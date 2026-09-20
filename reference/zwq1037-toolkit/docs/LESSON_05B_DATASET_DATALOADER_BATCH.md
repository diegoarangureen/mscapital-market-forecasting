# 第5B课：Dataset、DataLoader与batch

## 1. 本课目标

第5A课中，我们把4个样本一次性转换成Tensor并移动到了GPU。正式训练有125万多个样本，全部一次送进GPU通常会占用过多显存。

本课解决的问题是：

> 怎样保持每行特征与对应target不分离，并把大数据分成一小批一小批交给模型？

完成后，你应该能够：

- 用TensorDataset将features和targets按行配对；
- 用DataLoader按batch_size产生小批量；
- 使用for循环和解包取得batch_features与batch_targets；
- 预测最后一个不完整batch的shape；
- 解释训练集shuffle与时间验证泄漏的边界；
- 解释为什么通常在循环内只把当前batch送入GPU。

本课仍不建立神经网络，不计算loss，也不做反向传播。

## 2. 配套阅读

### 课前阅读：《动手学深度学习》第二版（PyTorch中文版）

- 本地文件：references/books/D2L_2nd_PyTorch_CN.pdf
- 章节：第3章，3.3.2“读取数据集”
- 书本页：101
- PDF页：119
- 阅读深度：精读这一页中的TensorDataset、DataLoader和第一个batch
- 暂时跳过：3.3.3之后的模型、参数初始化、损失和优化器；这些会在后续课程逐个学习

读后问题：

1. Dataset与DataLoader分别负责什么？
2. batch_size=10表示总数据只有10个样本吗？

书中使用了一个封装函数load_array。当前课程先直接调用两个核心类，避免函数封装遮住内部步骤。

## 3. 为什么需要batch

假设训练集有100万个样本。存在两种极端：

~~~text
一次使用全部100万个样本：
显存压力大，每次参数更新等待很久

一次只使用1个样本：
更新非常频繁，但GPU并行能力没有充分利用，梯度波动也大
~~~

通常把若干样本组成一个小批量 mini-batch：

~~~text
100万个样本
→ 每256个组成一个batch
→ 模型处理一个batch并更新一次
→ 继续下一个batch
~~~

batch_size只表示“每批最多取多少样本”，不是数据集总样本数。

若有5个样本，batch_size=2且不丢弃最后一批：

~~~text
第1批：2个样本
第2批：2个样本
第3批：1个样本
~~~

因此一共是3个batch，最后一个batch可以比batch_size小。

## 4. TensorDataset：保持特征与target按行配对

### 它解决什么问题

如果单独打乱features和targets，标签就可能对应错样本。TensorDataset把多个Tensor按第一维的位置绑定在一起。

### 来源与导入

TensorDataset是torch.utils.data模块中的类：

~~~python
from torch.utils.data import TensorDataset
~~~

### 基本调用语法

~~~python
dataset = TensorDataset(features, targets)
~~~

### 主要输入

可以传入一个或多个Tensor，但它们第一维长度必须相同。

例如：

~~~text
features.shape = (5, 3)
targets.shape  = (5, 1)
第一维都是5，因此可以配对
~~~

如果features有5行而targets只有4行，TensorDataset无法知道第5个样本的标签，会报尺寸不一致错误。

### 返回值

返回TensorDataset对象。它不复制出五份新表，也不自动产生batch；它只定义“第i个样本由哪些Tensor的第i行组成”。

### 最小例子

~~~python
import torch
from torch.utils.data import TensorDataset

features = torch.tensor(
    [[1.0, 2.0], [3.0, 4.0]],
    dtype=torch.float32,
)
targets = torch.tensor(
    [[0.1], [0.2]],
    dtype=torch.float32,
)

dataset = TensorDataset(features, targets)
~~~

访问第0个样本：

~~~python
sample_features, sample_target = dataset[0]
~~~

返回的是一个tuple，其中包含第0行特征和第0行target。Python从0开始编号。

### 当前比赛中的作用

未来训练MLP时，Dataset确保每个sample_id聚合出的62个特征始终与它自己的target一起被读取。

### 常见错误

TensorDataset按照位置配对，不认识sample_id。把DataFrame转Tensor之前，仍需先在pandas阶段完成按sample_id的一对一合并和顺序检查。

## 5. DataLoader：按规则从Dataset取batch

### 它解决什么问题

DataLoader负责：

- 每次从Dataset取多少个样本；
- 是否打乱顺序；
- 怎样连续产生多个batch；
- 后期还可使用多进程加载数据。

### 来源与导入

~~~python
from torch.utils.data import DataLoader
~~~

### 基本调用语法

~~~python
data_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
)
~~~

### 主要参数

- dataset：要读取的Dataset对象；
- batch_size：Python整数，每批最多包含多少个样本；
- shuffle：Python bool，是否在开始一轮读取时重新打乱样本顺序。

### 返回值

返回DataLoader对象。它是可迭代对象，可以放在for循环的in后面。循环每次返回一个batch。

### 常见错误

DataLoader不会自动把batch移动到GPU。Dataset中的Tensor如果位于CPU，取出的batch也默认位于CPU，需要在训练循环中调用to(device)。

## 6. for循环、解包与迭代器复习

基本写法：

~~~python
for batch_features, batch_targets in data_loader:
    print(batch_features.shape)
    print(batch_targets.shape)
~~~

逐部分理解：

- data_loader每次产生一个包含两个Tensor的tuple；
- batch_features接收tuple的第0项；
- batch_targets接收tuple的第1项；
- 这种一次给多个变量赋值的写法叫解包 unpacking；
- 缩进的代码块会对每个batch执行一次。

它等价于更拆开的思路：

~~~python
for batch in data_loader:
    batch_features = batch[0]
    batch_targets = batch[1]
~~~

我们使用解包是因为它清楚表达“每个batch包含特征和target”。

### batch_number += 1

~~~python
batch_number += 1
~~~

是下面写法的简写：

~~~python
batch_number = batch_number + 1
~~~

它用于记录当前已经处理到第几个batch。

## 7. 为什么在循环内移动batch，而不是先把全部数据送进GPU

第5A的小数据可以整体移动：

~~~python
features = features.to(device)
~~~

正式数据更常见的流程是：

~~~text
完整Dataset保留在CPU内存
→ DataLoader取出一个小batch
→ 只把这个batch送进GPU
→ 模型计算完再取下一批
~~~

代码结构是：

~~~python
for batch_features, batch_targets in data_loader:
    batch_features = batch_features.to(device)
    batch_targets = batch_targets.to(device)
~~~

这样GPU只需容纳当前batch和模型，不必一次容纳完整训练集。

## 8. shuffle会不会破坏金融时间验证

必须区分两个层面。

### 数据集之间的时间划分不能打乱

我们的底线仍然是：

~~~text
训练集：month 0～59
验证集：month 60～70
~~~

不能先把全部月份混在一起再随机拆分，否则未来样本可能进入训练集。

### 已划好的训练集内部可以shuffle

当训练集和验证集已经按月份分开后，DataLoader只接收训练集。对训练集内部样本使用shuffle=True，并不会把验证月份搬进训练集。

对于当前聚合特征MLP，每个sample_id被当作独立训练样本，训练batch顺序不代表模型可见的时间序列。因此通常：

~~~text
train DataLoader：shuffle=True
valid DataLoader：shuffle=False
~~~

后面若模型的隐藏状态跨样本连续传递，或使用严格的在线更新方案，是否shuffle需要重新判断，不能机械套用。

## 9. 最小完整例子

~~~python
import torch
from torch.utils.data import DataLoader, TensorDataset

# 三个样本，每个样本两个特征。
# Three samples, each with two features.
features = torch.tensor(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    dtype=torch.float32,
)
targets = torch.tensor(
    [[0.1], [0.2], [0.3]],
    dtype=torch.float32,
)

# 按第一维位置将特征和target配成三个样本。
# Pair features and targets by their first-dimension positions.
dataset = TensorDataset(features, targets)

# 每批最多读取两个样本，不打乱以便观察顺序。
# Read at most two samples per batch without shuffling.
data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch_features, batch_targets in data_loader:
    print(batch_features.shape, batch_targets.shape)
~~~

输出：

~~~text
torch.Size([2, 2]) torch.Size([2, 1])
torch.Size([1, 2]) torch.Size([1, 1])
~~~

## 10. 你的动手任务

打开exercises/lesson_05b_dataset_dataloader.py。

亲手完成：

1. 创建TensorDataset；
2. 创建batch_size=2、shuffle=False的DataLoader；
3. 补全for循环的数据来源；
4. 在循环中把当前batch的features和targets移动到device。

运行前先预测：

- 一共有几个batch；
- 三个features batch的shape；
- 三个targets batch的shape。

验收标准：5个样本应被完整读取且没有重复或漏掉；batch大小依次为2、2、1；所有batch进入循环后都位于所选device。

最后判断：

- A. 最后一个batch只有1个样本是正常情况。
- B. shuffle=True会分别打乱features和targets，因此一定造成错配。
- C. 先完成按月份的train/valid划分，再只打乱训练集内部的独立样本，不会把验证月份泄漏进训练集。

其中两个合理，一个大概率错误。完成后告诉Codex“已完成5B”。

## 11. 人话总结

~~~text
Tensor：保存数值
Dataset：定义一个样本由哪些对应数据组成
DataLoader：决定怎样成批取样本
batch：一次交给模型的一小组样本
to(device)：只把当前batch送到计算设备
~~~

下一课将处理神经网络比树模型更敏感的两个数据问题：特征尺度和NaN。
