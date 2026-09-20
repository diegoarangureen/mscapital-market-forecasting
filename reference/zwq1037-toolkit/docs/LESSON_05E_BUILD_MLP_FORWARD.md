# 第5E课：用nn.Sequential搭建MLP并完成前向传播

## 1. 本课目标

第5C课已经手算过：

~~~text
输入 → 仿射变换 → ReLU → 仿射变换 → 回归预测
~~~

本课把这条数学链真正写成PyTorch模型：

~~~text
3个输入特征
→ 4个隐藏神经元
→ ReLU
→ 1个连续输出
~~~

完成后，你应该能够：

- 使用nn.Linear建立全连接层；
- 使用nn.ReLU加入非线性；
- 使用nn.Sequential按顺序组合网络层；
- 把整个模型移动到CPU或CUDA；
- 调用model(batch_features)完成前向传播；
- 根据层宽推断输入、隐藏层、输出和参数shape；
- 解释为什么未经训练的预测数值没有评价意义。

本课不计算loss、不调用backward、不更新权重。先确保模型结构和数据流正确。

## 2. 配套阅读

### 课前阅读：《动手学深度学习》第二版（PyTorch中文版）

- 本地文件：references/books/D2L_2nd_PyTorch_CN.pdf
- 章节：第4章，4.3.1“模型”
- 书本页：138
- PDF页：156
- 精读：nn.Sequential、两个nn.Linear和中间nn.ReLU的排列
- 只理解结构：输入宽度、隐藏宽度、输出宽度
- 暂时跳过：Flatten、Fashion-MNIST的784输入/10分类、init_weights、CrossEntropyLoss、SGD和训练代码

读后问题：

1. 为什么ReLU放在隐藏层Linear之后、输出层Linear之前？
2. 如果最后一层有10个神经元，模型对每个样本会输出几个数？

书中的任务是图像分类，我们当前是金融回归，因此网络宽度和输出处理不能直接照抄。

## 3. from torch import nn是什么

~~~python
from torch import nn
~~~

这里从PyTorch的torch包导入nn模块。nn是neural network的缩写，里面提供：

- 神经网络层；
- 激活函数模块；
- 损失函数；
- 模型容器。

导入后使用点号访问其中的类：

~~~python
nn.Linear
nn.ReLU
nn.Sequential
~~~

这些名字本身是类；加圆括号才会创建具体对象。

## 4. nn.Linear：全连接仿射层

### 它解决什么问题

nn.Linear建立一个可学习的全连接层，计算：

$$
y=xW^T+b
$$

它自动创建并保存权重与偏置；以后optimizer会更新这些参数。

### 来源与基本语法

~~~python
layer = nn.Linear(
    in_features=3,
    out_features=4,
)
~~~

### 主要参数

- in_features：每个样本输入多少个数；
- out_features：每个样本输出多少个数，也就是这一层有多少个神经元；
- bias：是否使用偏置，默认为True。

### 输入与返回shape

输入：

$$
(batch\_size,in\_features)
$$

输出：

$$
(batch\_size,out\_features)
$$

例如：

~~~text
输入shape：(2, 3)
Linear(3, 4)
输出shape：(2, 4)
~~~

第一维2个样本不变，第二维从3个输入特征变成4个神经元输出。

### PyTorch权重shape为何与教材数学写法相反

第5C数学写作：

$$
XW,qquad W.shape=(3,4)
$$

PyTorch内部计算$xW^T+b$，因此保存：

~~~text
layer.weight.shape = (out_features, in_features) = (4, 3)
layer.bias.shape   = (out_features,)             = (4,)
~~~

两者描述同一个计算，只是保存方向不同。不要把PyTorch的$(4,3)$误认为4个输入、3个输出。

### 返回值与参数

nn.Linear(...)返回nn.Linear对象。创建时权重和偏置已经存在，但初始值是自动随机初始化的，还没有从数据学到规律。

### 当前比赛中的作用

正式MLP第一层可能是：

~~~python
nn.Linear(62, 64)
~~~

表示每个样本的62个聚合特征被组合成64个隐藏值。

### 常见错误

in_features必须等于实际输入Tensor的最后一维。若输入shape为$(32,62)$却使用nn.Linear(40,64)，矩阵尺寸无法相乘，会报shape错误。

## 5. nn.ReLU：可放进模型的激活层

第5C学过：

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

### 创建与调用

~~~python
activation = nn.ReLU()
result = activation(input_tensor)
~~~

nn.ReLU()返回一个ReLU模块对象。输入和返回都是Tensor，shape不变，只改变元素值：负数变0，正数保留。

### 为什么使用nn.ReLU()而不是手算

把它做成nn模块后，可以自然放入Sequential，并在前向传播时自动按顺序调用。

### 常见错误

如果两个Linear之间没有ReLU或其他非线性激活，两层整体仍可合并成一个仿射变换。

## 6. nn.Sequential：把层排成流水线

### 它解决什么问题

nn.Sequential按传入顺序保存多个模块。收到输入后，它把前一个模块的输出自动交给下一个模块。

### 基本语法

~~~python
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)
~~~

### 输入和返回值

nn.Sequential接收若干nn.Module对象，返回一个Sequential模型对象。

调用模型时：

~~~python
predictions = model(batch_features)
~~~

内部相当于：

~~~python
hidden_before_relu = model[0](batch_features)
hidden_after_relu = model[1](hidden_before_relu)
predictions = model[2](hidden_after_relu)
~~~

紧凑写法负责运行，分步写法负责帮助理解。

### 当前shape链条

若batch_features.shape为$(2,3)$：

~~~text
(2, 3)
→ Linear(3, 4)
→ (2, 4)
→ ReLU
→ (2, 4)
→ Linear(4, 1)
→ (2, 1)
~~~

### 常见错误

相邻层宽度必须衔接：

~~~text
Linear(3, 4)的输出是4
所以下一层必须接收4，例如Linear(4, 1)
~~~

写成Linear(5,1)会产生矩阵shape错误。

## 7. 为什么输出层后不加ReLU

当前比赛target可能为正，也可能为负。

若输出层后加ReLU：

$$
\hat y=\max(0,z)
$$

所有负预测都会被强制变成0，模型无法输出负target。因此第一版回归MLP使用：

~~~python
nn.Linear(4, 1)
~~~

作为最后一层，不在它后面加ReLU或sigmoid。

隐藏层ReLU和输出层是否使用激活函数是两个不同决定：

~~~text
隐藏层ReLU：提供非线性表达能力
输出层无激活：允许连续预测覆盖正数和负数
~~~

## 8. model.to(device)：移动全部模型参数

第5A已经使用Tensor的to(device)。模型同样需要移动：

~~~python
model = model.to(device)
~~~

它会递归移动Sequential中每个Linear的weight和bias。网络结构和参数shape不变，只改变存储设备。

必须保证：

~~~text
model参数device
batch_features.device
batch_targets.device
~~~

三者一致。当前只做前向传播时targets暂时不参与计算，但正式训练时也必须移动。

## 9. model(batch_features)是怎样调用模型的

~~~python
predictions = model(batch_features)
~~~

model是对象，圆括号表示调用该对象。PyTorch的nn.Module实现了可调用行为，它会进入模型的forward流程，并处理框架需要的内部机制。

对于Sequential，forward就是按照保存顺序调用所有子模块。

不要直接写：

~~~python
predictions = model.forward(batch_features)
~~~

通常应调用model(...)，让PyTorch保留hooks等框架行为。

返回predictions是Tensor。当前模型最后一层输出宽度为1，所以：

$$
predictions.shape=(batch\_size,1)
$$

## 10. 为什么未经训练的预测没有意义

刚创建模型时：

~~~text
Linear权重：自动随机初始化
bias：按PyTorch默认规则初始化
~~~

因此model(batch_features)能够产生合法数值和正确shape，但它还没有根据target调整权重。

本课只检查：

- 程序能否前向运行；
- shape是否正确；
- dtype和device是否一致；
- 输出层是否允许负数。

不能用本课随机预测判断模型效果。下一课才会引入loss与自动求导。

## 11. 查看模型与参数shape

打印模型：

~~~python
print(model)
~~~

输出会显示Sequential中的层顺序。

模型方法named_parameters()会返回一个可迭代对象，每次提供：

~~~text
(参数名称, 参数Tensor)
~~~

基本写法：

~~~python
for parameter_name, parameter in model.named_parameters():
    print(parameter_name, parameter.shape)
~~~

这里使用了第5B复习过的for循环与解包。对于Linear(3,4)和Linear(4,1)，预期参数shape是：

~~~text
0.weight  (4, 3)
0.bias    (4,)
2.weight  (1, 4)
2.bias    (1,)
~~~

0和2是这些模块在Sequential中的位置；位置1是ReLU，它没有可学习权重或偏置，所以不会出现在named_parameters()中。

## 12. 你的动手任务

打开exercises/lesson_05e_build_mlp_forward.py。

固定Tensor、Dataset、DataLoader和设备选择已经提供。你亲手完成：

1. 用Sequential搭建Linear(3,4) → ReLU → Linear(4,1)；
2. 把model移动到device；
3. 在循环中调用model(batch_features)得到predictions。

运行前预测：

- 三个batch的predictions shape；
- 第一层与最后一层的weight shape；
- ReLU是否拥有weight；
- 为什么最后一层后面不加ReLU。

判断题：

- A. Linear(3,4)接收每个样本3个特征，并为每个样本输出4个隐藏值。
- B. 只要前向传播能运行，随机初始化模型的预测数值就已经可以评价模型效果。
- C. 当前输出层不加ReLU，是因为回归target和预测都需要允许负数。

其中两个合理，一个大概率错误。完成后告诉Codex“已完成5E”。

## 13. 人话总结

~~~text
nn.Linear：保存并执行可学习的权重与偏置
nn.ReLU：给隐藏表示加入非线性
nn.Sequential：按顺序串联多个模块
model.to(device)：移动所有模型参数
model(batch)：使用当前参数完成前向传播
未经训练：只检查结构，不能评价效果
~~~
