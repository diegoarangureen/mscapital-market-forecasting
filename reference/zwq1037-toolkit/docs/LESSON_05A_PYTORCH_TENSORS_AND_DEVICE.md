# 第5A课：PyTorch Tensor、shape、dtype与device

## 1. 本课在整个项目中的位置

我们已经完成阶段4的LightGBM基线与近期窗口实验。现在进入阶段5：PyTorch与MLP过渡。

LightGBM可以直接接收一张表；神经网络则要求输入变成PyTorch Tensor，并且要求数据、模型和计算设备彼此匹配。本课先解决这个最基础的问题，不训练模型。

完成本课后，你应该能够独立解释并完成：

- 把二维Python列表转换为Tensor；
- 从shape判断样本数和特征数；
- 区分float32数据类型和CPU/CUDA设备；
- 把特征和target移动到同一个计算设备；
- 解释为什么本项目未来的MLP输入大致是“样本数 × 特征数”。

本课暂时不学习：Dataset、DataLoader、batch、神经网络层、loss、反向传播和优化器。这些会分到后续课程，避免一次引入过多对象。

## 2. 配套阅读

### 课前阅读：《动手学深度学习》第二版（PyTorch中文版）

- 本地文件：references/books/D2L_2nd_PyTorch_CN.pdf
- 章节：第2章“预备知识”，2.1“数据操作”中的2.1.1～2.1.4
- 书本页：40～45
- PDF页：58～63
- 阅读深度：理解直觉并运行时对照，不要求背API
- 重点：张量的创建、shape、基本运算、广播、索引和切片
- 暂时跳过：2.1.5“节省内存”、2.1.6“转换为其他Python对象”，以及本章后续微积分与自动微分

读后只回答两个问题：

1. 二维Tensor的两个维度分别可以表示什么？
2. 为什么两个shape不同的Tensor不一定能直接进行逐元素运算？

### 课后选读：第5章“深度学习计算”中的5.6 GPU

- 书本页：211～215
- PDF页：229～233
- 阅读时机：完成本课练习以后
- 阅读深度：只理解“Tensor位于某个设备”和“参与同一次运算的对象应位于同一设备”
- 暂时跳过：多GPU训练和任何性能比较

PyTorch接口的具体用法由本讲义解释，不要求你阅读官方API文档。

## 3. 为什么不能直接把DataFrame交给神经网络

pandas DataFrame擅长：

- 保存带列名的表格；
- 清洗、筛选、分组和聚合；
- 观察缺失值和统计信息。

PyTorch Tensor擅长：

- 进行大量数值运算；
- 在GPU上计算；
- 记录自动求导所需的信息；
- 作为神经网络层的标准输入与输出。

因此，后续流程大致是：

~~~text
pandas特征表
→ 处理缺失和数值尺度
→ 转换为Tensor
→ 组成batch
→ 输入神经网络
~~~

Tensor可以理解成“为深度学习计算准备的多维数值数组”。

## 4. torch.tensor()：从现有数据创建Tensor

### 它解决什么问题

torch.tensor()把Python列表、NumPy数组等现有数据复制为PyTorch Tensor。

### 来源与导入

它来自PyTorch：

~~~python
import torch
~~~

### 基本调用语法

~~~python
result = torch.tensor(data, dtype=torch.float32)
~~~

### 主要输入

- data：Python列表、嵌套列表或NumPy数组；
- dtype：希望Tensor使用的数据类型。

### 返回值

返回torch.Tensor对象。返回结果的shape取决于输入数据的嵌套结构。

### 最小例子

~~~python
import torch

# 两行表示两个样本，每行三个数表示三个特征。
# Two rows represent two samples, with three features in each row.
feature_rows = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
]

# 将二维列表转换为float32 Tensor。
# Convert the two-dimensional list into a float32 Tensor.
feature_tensor = torch.tensor(feature_rows, dtype=torch.float32)
~~~

结果的shape是：

$$
(2,3)=(2\text{个样本},3\text{个特征})
$$

### 在当前比赛中的作用

我们的LightGBM V1 + last60特征共有62列。以后若一次把32个样本组成一个batch，MLP看到的特征Tensor将类似：

$$
X.shape=(32,62)
$$

target只有一个连续数值，因此通常是：

$$
y.shape=(32,1)
$$

### 常见错误

不要把特征创建成整数Tensor。神经网络权重通常是float32；输入若是整数类型，进行线性层计算时会出现数据类型不匹配。

## 5. shape、dtype和device是三个不同问题

它们都是Tensor对象的属性，所以使用点号访问，不加圆括号：

~~~python
print(feature_tensor.shape)
print(feature_tensor.dtype)
print(feature_tensor.device)
~~~

### shape：数据怎样排列

shape的类型是torch.Size，行为很像一个只读的尺寸序列。

~~~text
shape = (4, 3)
         ↑  ↑
       4个样本
          3个特征
~~~

shape只描述排列方式，不告诉你数据位于CPU还是GPU。

### dtype：每个数字怎样存储

dtype表示数据类型。当前阶段最常用的是：

~~~text
torch.float32：神经网络常用浮点数
torch.float64：精度更高，但内存和计算成本更高
torch.int64：常用于类别编号，不适合作为当前回归特征的默认类型
~~~

本项目的特征和连续target都先统一为torch.float32。

### device：数据在哪里计算

常见输出是：

~~~text
cpu       数据在CPU内存中
cuda:0    数据在第1块NVIDIA GPU上
~~~

同一次运算中的Tensor和模型参数必须位于同一设备。CPU Tensor不能直接与CUDA Tensor做矩阵运算。

## 6. torch.cuda.is_available()：CUDA现在能不能用

### 它解决什么问题

它检查当前PyTorch安装、显卡和驱动是否共同提供可用的CUDA环境。

### 来源和语法

~~~python
cuda_available = torch.cuda.is_available()
~~~

它不接收参数，返回Python bool：True或False。

注意：电脑装有NVIDIA显卡，不等于当前PyTorch一定能使用CUDA。CPU版PyTorch、驱动问题或环境选错都可能让结果为False。

## 7. torch.device与if/else设备选择

torch.device是PyTorch中的类，用来创建“设备说明对象”：

~~~python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
~~~

这里的if/else是Python条件语句：条件为True时执行第一个缩进代码块，否则执行else代码块。

输入是字符串"cuda"或"cpu"，返回torch.device对象。它只是说明以后准备使用哪个设备，并不会自动移动已经创建的Tensor。

## 8. tensor.to(device)：移动或转换Tensor

### 它解决什么问题

to()是Tensor对象的方法，用于把Tensor移动到指定设备，也可以转换dtype。

### 基本语法

~~~python
moved_tensor = original_tensor.to(device)
~~~

### 输入与返回值

- 输入device：torch.device对象或设备字符串；
- 返回值：位于目标设备上的Tensor；
- shape保持不变。

### 最小例子

~~~python
# 将特征移动到选择好的设备，并接住返回值。
# Move the features to the selected device and keep the returned tensor.
feature_tensor_on_device = feature_tensor.to(device)
~~~

### 当前比赛中的作用

以后每个batch都需要把X和y移到模型所在设备：

~~~text
model → device
X batch → device
y batch → device
~~~

三者缺一不可。

### 常见错误

下面这行不应被理解成“一定原地修改成功”：

~~~python
feature_tensor.to(device)
~~~

安全、清楚的写法是接住返回值：

~~~python
feature_tensor = feature_tensor.to(device)
~~~

## 9. 本课最小完整例子

~~~python
import torch

# 两个样本，每个样本有三个输入特征。
# Two samples, each with three input features.
feature_rows = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
]

# 每个样本对应一个连续回归目标。
# Each sample has one continuous regression target.
target_rows = [
    [0.2],
    [-0.1],
]

# 创建float32 Tensor，供后续神经网络使用。
# Create float32 tensors for later neural-network computation.
features = torch.tensor(feature_rows, dtype=torch.float32)
targets = torch.tensor(target_rows, dtype=torch.float32)

# 根据当前环境选择CUDA或CPU。
# Select CUDA or CPU according to the current environment.
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 将输入和目标移动到同一设备。
# Move inputs and targets to the same device.
features = features.to(device)
targets = targets.to(device)

print("features shape =", features.shape)
print("targets shape =", targets.shape)
print("features dtype =", features.dtype)
print("features device =", features.device)
~~~

预期shape：

~~~text
features shape = torch.Size([2, 3])
targets shape = torch.Size([2, 1])
~~~

具体device取决于当前Conda环境和CUDA状态，不强制必须是CUDA。

## 10. 你的动手任务

打开exercises/lesson_05a_tensors_and_device.py。

你只需要亲手完成三个有学习价值的TODO：

1. 用torch.tensor()创建float32特征Tensor；
2. 用torch.tensor()创建float32 target Tensor；
3. 用to(device)将二者移动到已经选择好的设备。

路径、固定小数据、设备判断和打印代码已经提供，不需要机械重写。

运行前，先在文件底部写下三个预测：

- features的shape；
- targets的shape；
- 两者的dtype。

验收标准：

~~~text
features shape = torch.Size([4, 3])
targets shape = torch.Size([4, 1])
两者dtype都是torch.float32
两者device都与selected device相同
~~~

运行后再回答一个判断题：

- A. features的第一维4表示样本数，第二维3表示每个样本的特征数。
- B. 即使本机CUDA可用，刚由torch.tensor()创建的Tensor默认仍在CPU，除非显式指定或移动设备。
- C. 只把模型移动到CUDA即可，输入Tensor留在CPU也能直接完成同一次前向计算。

其中两个合理，一个大概率错误。选出错误项并说明原因。

完成后告诉Codex“已完成5A”。Codex会读取并运行你的文件，不直接代改核心TODO。

## 11. 人话总结

~~~text
Tensor是什么：神经网络使用的多维数值数组
shape是什么：数据各维度有多长
dtype是什么：每个数字用什么类型保存
device是什么：数据位于CPU还是GPU
to(device)做什么：把Tensor移动到目标设备
~~~

下一课才会学习Dataset与DataLoader怎样把大量样本按batch送给模型。
