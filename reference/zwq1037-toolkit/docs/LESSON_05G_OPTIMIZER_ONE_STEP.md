# 第5G课：optimizer、zero_grad与一次参数更新

## 1. 本课目标

第5F结束时，每个参数的grad已经存在，但weight和bias尚未改变。本课让模型真正学习一步：

~~~text
清除旧梯度
→ 前向传播
→ 计算loss
→ backward计算梯度
→ optimizer.step更新参数
→ 重新前向计算新loss
~~~

完成后，你应该能够：

- 创建最基础的SGD优化器；
- 解释model.parameters()交给了优化器什么；
- 解释learning rate如何控制更新幅度；
- 正确排列zero_grad、forward、loss、backward和step；
- 比较step前后的参数；
- 解释为什么step后必须重新前向传播才能得到新loss；
- 解释为什么保存参数快照需要detach().clone()。

本课只更新一个batch一次，不写完整epoch循环，也不引入Adam。

## 2. 配套阅读

### 《动手学深度学习》第二版（PyTorch中文版）

- 本地文件：references/books/D2L_2nd_PyTorch_CN.pdf
- 章节：第3章，3.3.6“定义优化算法”与3.3.7“训练”
- 书本页：103～104
- PDF页：121～122
- 精读：SGD的创建方式，以及每个batch的前向、清梯度、反向和step顺序
- 只理解结构：epoch外层循环与batch内层循环
- 暂时跳过：教材的合成线性回归结果、真实参数误差和课后题

读后问题：

1. backward和step分别承担什么职责？
2. 为什么每个batch都需要清除上一次留下的梯度？

## 3. optimizer是什么

梯度只说明当前位置的坡度：

$$
\frac{\partial L}{\partial\theta}
$$

其中$\theta$代表某个weight或bias。优化器optimizer负责根据梯度和更新规则真正改变$\theta$。

本课使用最简单的随机梯度下降Stochastic Gradient Descent：

$$
\theta_{new}
=\theta_{old}-\eta\frac{\partial L}{\partial\theta}
$$

其中$\eta$是learning rate学习率。

人话解释：

- 梯度指向loss上升方向；
- 前面的负号让参数向loss下降方向移动；
- 学习率决定一步走多远。

## 4. torch.optim.SGD：创建SGD优化器

### 它解决什么问题

SGD根据每个参数的grad执行参数更新，不需要我们逐个weight手写减法。

### 来源与语法

SGD位于torch.optim模块：

~~~python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
)
~~~

### 主要输入

- model.parameters()：模型中所有需要训练的Parameter迭代器；
- lr：Python浮点数，学习率；
- 本课不设置momentum或weight_decay，以便更新严格对应最简单公式。

### model.parameters()是什么

这是模型对象的方法，返回一个可迭代对象，其中依次包含：

~~~text
第一层weight
第一层bias
最后一层weight
最后一层bias
~~~

优化器保存对这些Parameter的引用，因此step时可以读取它们的grad并修改它们的数值。

### 返回值

torch.optim.SGD(...)返回SGD优化器对象。创建优化器不会立刻计算梯度或修改权重。

### 常见错误

不要忘记把model.parameters()传进去。优化器必须知道它负责更新哪些参数。

## 5. learning rate控制什么

假设：

$$
\theta_{old}=1,qquad grad=-2
$$

若学习率0.1：

$$
\theta_{new}=1-0.1\times(-2)=1.2
$$

若学习率1：

$$
\theta_{new}=1-1\times(-2)=3
$$

两次方向相同，但第二次走得更远。

- learning rate太小：学习很慢；
- learning rate太大：可能越过低点、来回震荡，甚至让loss爆炸；
- 一次step后loss下降不代表该学习率长期一定最好。

## 6. optimizer.zero_grad()为什么需要

PyTorch默认把新梯度累加到已有grad：

~~~text
旧grad + 本batch新grad → parameter.grad
~~~

普通mini-batch训练希望每次更新只依据当前batch，因此在新一轮backward之前清除旧梯度：

~~~python
optimizer.zero_grad()
~~~

它是optimizer对象的方法，不需要输入参数，通常返回None。

现代PyTorch默认可能把grad设为None，而不是创建全0 Tensor。两种状态都表示旧梯度不参与下一次累积。

常见错误是漏掉zero_grad。结果不是“模型多学一点”，而是不同batch的梯度无计划地不断累积，更新幅度和含义都改变。

## 7. optimizer.step()做什么

~~~python
optimizer.step()
~~~

它读取各参数当前的parameter.grad，按照SGD规则修改weight和bias。

调用顺序必须是：

~~~text
先backward产生grad
→ 再step读取grad并更新参数
~~~

如果在backward前调用step，优化器没有本batch的新梯度可用。

step通常返回None。执行step以后，旧grad仍然可能保留在parameter.grad中；step不会替下一批自动清除梯度。

## 8. 标准的单batch训练顺序

~~~python
optimizer.zero_grad()

predictions = model(batch_features)
loss = loss_function(predictions, batch_targets)

loss.backward()
optimizer.step()
~~~

每一行的职责：

~~~text
zero_grad：清理上一批留下的梯度
model：使用当前参数产生预测
loss_function：衡量当前预测误差
backward：计算本batch对各参数的梯度
step：使用这些梯度改变参数
~~~

zero_grad放在forward之前或loss之后、backward之前通常都可以；关键底线是必须在本次backward前清掉不需要的旧梯度。为了形成稳定习惯，本项目统一放在batch开头。

## 9. 为什么step后旧loss不会自动变化

~~~python
loss = loss_function(model(batch_features), batch_targets)
loss.backward()
optimizer.step()
~~~

变量loss保存的是step之前那次前向传播产生的Tensor。参数改变以后，它不会自动重新执行过去的计算。

要观察更新后的损失，必须重新前向传播：

~~~python
predictions_after = model(batch_features)
loss_after = loss_function(predictions_after, batch_targets)
~~~

因此比较的是：

~~~text
旧参数 → forward → loss_before
step更新参数
新参数 → 重新forward → loss_after
~~~

## 10. 为什么参数快照需要detach().clone()

下面写法不能保存独立旧权重：

~~~python
weight_before = model[0].weight
~~~

这只是让weight_before和model[0].weight指向同一个Parameter对象。step修改Parameter后，从两个变量看到的都是新值。

安全快照：

~~~python
weight_before = model[0].weight.detach().clone()
~~~

### detach()

detach()是Tensor方法，返回与原Tensor共享数值存储、但脱离计算图的新Tensor视图。它不再要求梯度，适合检查和记录参数。

单独detach仍可能共享底层数据，所以还不足以冻结旧值。

### clone()

clone()是Tensor方法，复制Tensor数据，返回shape和dtype相同、但拥有独立存储的新Tensor。

连起来：

~~~text
detach：不把快照纳入梯度计算
clone：复制独立数据，避免step后一起变化
~~~

在本课中它只用于验证参数确实被更新，不参与模型训练。

## 11. 用一个参数检查SGD公式

更新前保存：

~~~python
weight_before = model[0].weight.detach().clone()
~~~

backward后读取某个梯度：

~~~python
gradient = model[0].weight.grad[0, 0].item()
~~~

step后读取新权重：

~~~python
weight_after = model[0].weight[0, 0].item()
~~~

它们应近似满足：

$$
weight\_after
=weight\_before-lr\times gradient
$$

浮点计算可能存在极小舍入差异，不要求打印小数完全逐位相同。

## 12. 当前比赛中先学SGD、后用Adam的原因

SGD的更新公式最直接，适合第一次观察梯度怎样改变参数。正式MLP常会尝试Adam，因为它会为不同参数维护自适应更新信息，通常更容易得到可用起点。

但Adam不会改变训练基本顺序：

~~~text
zero_grad → forward → loss → backward → step
~~~

所以先理解SGD不是浪费，后面只需替换optimizer对象。

## 13. 你的动手任务

打开exercises/lesson_05g_optimizer_one_step.py。

固定数据、模型、首个batch和旧权重快照已经提供。你亲手完成：

1. 创建lr=0.1且无momentum的SGD优化器；
2. 清除旧梯度；
3. 计算更新前预测和loss；
4. backward计算梯度；
5. step更新参数；
6. 用新参数重新前向并计算loss_after。

运行前预测：

- backward后、step前权重是否改变；
- step后权重是否改变；
- step会不会自动清除grad；
- 旧loss Tensor会不会随权重变化自动更新。

判断题：

- A. backward负责产生grad，step负责读取grad并更新参数。
- B. step之后必须重新前向传播，才能计算新参数对应的loss。
- C. weight_before=model[0].weight能够保存不会随step变化的独立旧权重副本。

其中两个合理，一个大概率错误。完成后告诉Codex“已完成5G”。

## 14. 人话总结

~~~text
grad：告诉参数附近的loss坡度
learning rate：决定沿反方向走多远
zero_grad：丢掉上一批不再需要的梯度
backward：计算本批梯度
step：真正改变参数
重新forward：得到新参数对应的新预测与loss
detach().clone()：保存不会跟着参数变化的旧值快照
~~~
