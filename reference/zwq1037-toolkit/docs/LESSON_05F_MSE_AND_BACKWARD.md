# 第5F课：MSELoss、计算图与一次backward

## 1. 本课目标

第5E的MLP已经能够输出预测，但模型还不知道预测错了多少，也没有任何学习发生。

本课只完成一个batch的这条链：

~~~text
batch_features
→ model前向传播
→ predictions
→ 与batch_targets比较
→ 得到一个标量loss
→ loss.backward()
→ 每个模型参数得到grad
~~~

完成后，你应该能够：

- 使用nn.MSELoss计算回归损失；
- 解释默认MSE为什么是一个0维标量Tensor；
- 区分loss Tensor与loss.item()返回的Python数字；
- 用人话解释计算图；
- 调用loss.backward()计算梯度；
- 解释parameter.grad为何与parameter具有相同shape；
- 明确backward不会自动更新模型权重。

本课不创建optimizer，也不写完整训练循环。

## 2. 配套阅读

### 阅读一：《动手学深度学习》第二版，第2.5节“自动微分”

- 本地文件：references/books/D2L_2nd_PyTorch_CN.pdf
- 范围：第2.5节开头和2.5.1“一个简单的例子”
- 书本页：69～70
- PDF页：87～88
- 精读：计算图、requires_grad、backward和grad
- 暂时跳过：2.5.2非标量反向传播、detach与Python控制流

### 阅读二：第3.3.5“定义损失函数”

- 书本页：103
- PDF页：121
- 只读MSELoss段落；跳过后面的SGD优化器

读后问题：

1. backward计算出的结果存在哪里？
2. backward执行后，权重是否已经改变？

## 3. MSELoss解决什么问题

模型产生预测：

$$
\hat y_1,\hat y_2,\ldots,\hat y_N
$$

但训练需要一个数衡量它们与真实target的整体差距。均方误差Mean Squared Error定义为：

$$
\operatorname{MSE}
=\frac1N\sum_{i=1}^{N}(\hat y_i-y_i)^2
$$

它进行三步：

~~~text
预测减真实值
→ 平方，让正负误差都成为非负并放大大误差
→ 对所有元素取平均
~~~

### 数字例子

$$
\hat y=[2,4],\qquad y=[1,3]
$$

那么：

$$
\operatorname{MSE}
=\frac{(2-1)^2+(4-3)^2}{2}
=\frac{1+1}{2}=1
$$

注意：第4F推导LightGBM时使用了$\frac12(y-\hat y)^2$方便求导；PyTorch的默认MSELoss没有额外的$\frac12$。两者最优点相同，但梯度会相差一个常数倍。

## 4. nn.MSELoss的来源与用法

### 来源

~~~python
from torch import nn
~~~

MSELoss是nn模块中的类。

### 创建损失函数对象

~~~python
loss_function = nn.MSELoss()
~~~

默认参数是reduction="mean"，表示把所有元素的平方误差取平均。

### 调用

~~~python
loss = loss_function(predictions, batch_targets)
~~~

主要输入：

- predictions：模型预测Tensor；
- batch_targets：真实目标Tensor；
- 两者shape应该相同，例如都为$(2,1)$。

返回值loss是0维Tensor：

~~~text
loss.shape = torch.Size([])
~~~

它只包含一个数，但仍然是Tensor，并且连接着计算图。

### 常见错误：预测与target shape不一致

例如：

~~~text
predictions.shape = (32, 1)
targets.shape     = (32,)
~~~

PyTorch可能尝试broadcasting，而不是立即按你的意图逐样本比较，从而产生错误损失。训练前必须明确检查两者shape完全相同。

## 5. 标量Tensor和Python数字的区别

loss可能显示为：

~~~text
tensor(0.0312, device='cuda:0', grad_fn=<MseLossBackward0>)
~~~

它是Tensor，可以：

~~~python
loss.backward()
~~~

为了记录日志，可以取出普通Python数字：

~~~python
loss_value = loss.item()
~~~

item()是Tensor对象的方法：

- 输入：不需要参数；
- 要求：Tensor中只能有一个元素；
- 返回：Python的float或int；
- 用途：打印、记录实验结果；
- 注意：返回的Python数字不再连接计算图，不能调用backward。

所以通常保留loss Tensor用于反向传播：

~~~python
loss.backward()
print(loss.item())
~~~

## 6. 什么是计算图

计算图不是我们手动画出的图片，而是PyTorch在前向计算时记录的依赖关系。

当前链条可以写成：

~~~text
Linear1的weight和bias ─┐
batch_features ─────────┼→ Linear → ReLU → Linear → predictions
Linear2的weight和bias ─┘                         │
batch_targets ───────────────────────────────────┼→ MSE → loss
~~~

PyTorch记录：

- predictions由哪些参数和运算产生；
- loss又由predictions怎样产生；
- 反向时梯度应沿哪些路径传回参数。

模型参数默认：

~~~text
parameter.requires_grad = True
~~~

普通输入features通常不需要求关于输入本身的梯度，所以默认requires_grad=False并没有问题。我们的目标是调整模型参数，而不是调整训练数据。

## 7. loss.backward()做什么

~~~python
loss.backward()
~~~

backward()是标量loss Tensor的方法。它从loss出发，按照链式法则沿计算图反向计算loss对每个可学习参数的偏导数。

调用之前：

~~~text
parameter.grad通常是None
~~~

调用之后：

~~~text
parameter.grad是一个Tensor
parameter.grad.shape与parameter.shape相同
~~~

例如第一层：

~~~text
0.weight.shape      = (4, 3)
0.weight.grad.shape = (4, 3)
~~~

因为每一个权重都需要知道：“如果我稍微增加，loss会怎样变化？”所以每个参数元素都有一个对应梯度。

## 8. 一个梯度的数字直觉

只有一个权重的模型：

$$
\hat y=wx
$$

给定：

$$
w=1,\quad x=2,\quad y=5
$$

预测：

$$
\hat y=1\times2=2
$$

单样本MSE：

$$
L=(2-5)^2=9
$$

对$w$求导：

$$
\frac{\partial L}{\partial w}
=2(\hat y-y)x
=2(2-5)\times2=-12
$$

梯度为负表示：在当前位置增加$w$会让loss下降。下一课的optimizer会利用这个方向更新$w$。

## 9. backward不会做什么

loss.backward()只负责：

~~~text
计算梯度
→ 把梯度累积到parameter.grad
~~~

它不会：

- 改变weight或bias；
- 决定learning rate；
- 清除上一批梯度；
- 自动开始下一个batch。

真正更新参数需要optimizer.step()。清除旧梯度需要optimizer.zero_grad()。这两个接口放到5G学习。

特别注意：PyTorch默认累积梯度。如果不清零就对下一个batch继续backward，新梯度会加到旧grad上，而不是覆盖。现在只做一次backward，因此本课暂时不调用zero_grad。

## 10. 为什么loss通常要是标量

默认MSELoss把batch内误差取平均，得到一个标量目标：

$$
L\in\mathbb R
$$

然后backward计算这个整体目标对全部参数的梯度。

如果loss包含多个元素，PyTorch需要知道这些元素应怎样组合后再求导。实际训练通常先用mean或sum得到标量，避免歧义。

## 11. 当前比赛中MSE与cosine的分工

第一版MLP可以继续采用：

~~~text
训练objective：MSELoss
验证metric：cosine similarity
~~~

MSELoss产生平滑、容易优化的逐样本误差信号，用来计算梯度。cosine用于判断整段验证预测的方向是否与真实target一致。

与LightGBM相同：训练损失改善不保证验证cosine一定改善，因此以后仍要保存最佳验证轮次。

## 12. 你的动手任务

打开exercises/lesson_05f_mse_backward.py。

固定数据、MLP、第一个batch和设备移动已经提供。你亲手完成：

1. 创建nn.MSELoss对象；
2. 用模型得到predictions；
3. 用predictions和targets得到loss；
4. 调用loss.backward()。

运行前预测：

- predictions、targets和loss的shape；
- backward之前parameter.grad是什么；
- backward之后grad的shape；
- backward是否已经改变权重。

判断题：

- A. loss.backward()计算并保存梯度，但不会自行更新权重。
- B. 每个parameter.grad的shape应该与对应parameter相同。
- C. loss.item()返回的Python浮点数仍保留计算图，因此可以继续调用backward()。

其中两个合理，一个大概率错误。完成后告诉Codex“已完成5F”。

## 13. 人话总结

~~~text
MSELoss：把预测误差汇总成一个训练目标
计算图：记录loss是怎样由模型参数计算出来的
backward：沿计算图反向计算每个参数的梯度
parameter.grad：保存梯度
item：只取日志数字，丢掉计算图
backward之后：有梯度，但权重还没更新
~~~
