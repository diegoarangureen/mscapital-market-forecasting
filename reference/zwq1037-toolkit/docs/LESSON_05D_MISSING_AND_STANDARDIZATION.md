# 第5D课：MLP输入的缺失值与特征标准化

## 1. 本课目标

LightGBM可以直接利用NaN，并且对特征数值尺度不太敏感。MLP不同：

- NaN进入神经网络后会在矩阵运算、预测、loss和梯度中继续传播；
- 不同特征的数值范围相差巨大，会让梯度优化变得困难。

当前62个聚合特征中，有些是接近1的价格，有些是成千上万的盘口量，有些近期窗口特征还会因为窗口内没有记录而出现NaN。本课为MLP建立安全输入。

完成后，你应该能够：

- 解释NaN为什么不能直接送入MLP；
- 使用训练集统计量填充训练集与验证集；
- 手算标准化公式；
- 使用StandardScaler只在训练集fit；
- 解释标准化后验证值为什么可能大于1或小于-1；
- 把处理后的数组转换成float32 Tensor。

本课不建立MLP，也不训练模型。每次实验只增加一个主要难点。

## 2. 配套阅读

### 课前阅读：《动手学深度学习》第二版（PyTorch中文版）

- 本地文件：references/books/D2L_2nd_PyTorch_CN.pdf
- 章节：第4章，4.10.4“数据预处理”
- 书本页：184～185
- PDF页：202～203
- 精读：均值/标准差标准化公式、缺失值处理、转换为float32 Tensor
- 只浏览：独热编码；当前62个聚合特征已经是数值特征
- 暂时跳过：4.10.5训练及后续K折代码

### 教材示例的使用边界

教材为了房价比赛示范方便，先合并train和test再计算预处理统计量。我们不复制这一做法。

当前项目要模拟“用过去预测未来”，因此：

~~~text
训练月份0～59：允许fit填充值、均值和标准差
验证月份60～70：只能使用训练期学到的统计量transform
Kaggle测试月份71～108：同样只能transform
~~~

读后问题：

1. 标准化公式中的均值和标准差应该从哪部分数据计算？
2. 为什么把NaN直接转换为Tensor并不能自动解决缺失问题？

## 3. NaN进入MLP会发生什么

一个神经元计算：

$$
z=w_1x_1+w_2x_2+b
$$

如果$x_2=\mathrm{NaN}$：

$$
w_2\times\mathrm{NaN}=\mathrm{NaN}
$$

所以：

$$
z=\mathrm{NaN}
$$

随后ReLU、后续层和loss也很可能变成NaN：

~~~text
一个输入NaN
→ 隐藏值NaN
→ 预测NaN
→ loss NaN
→ 梯度NaN
→ 权重被破坏
~~~

“神经元失效”只是方便理解的说法。更准确地说，是NaN通过数值运算传播，最终让训练无法产生有效参数更新。

## 4. SimpleImputer复习：先填充缺失值

SimpleImputer来自scikit-learn：

~~~python
from sklearn.impute import SimpleImputer
~~~

本课采用训练列中位数：

~~~python
imputer = SimpleImputer(strategy="median")
~~~

训练集既学习中位数又完成转换：

~~~python
X_train_imputed = imputer.fit_transform(X_train)
~~~

验证集只使用已学到的训练中位数：

~~~python
X_valid_imputed = imputer.transform(X_valid)
~~~

输入通常是二维DataFrame或NumPy数组，返回二维NumPy数组，行数和列数通常保持不变。

### 数字例子

训练集某列为：

~~~text
[1, NaN, 3, 5]
~~~

非缺失值为$[1,3,5]$，训练中位数是3。训练和验证中该列的NaN都填为3。

如果验证列是：

~~~text
[NaN, 100]
~~~

不能使用验证集自己的中位数100填充，因为这会让预处理利用未来验证数据的分布。仍然填训练中位数3。

### 缺失本身可能是信息

填值解决的是数值计算安全，不代表缺失含义应被抹掉。本项目已经保留了例如：

- has_transaction_ratio；
- transaction_count_sum；
- last60_row_count。

这些特征帮助模型区分“真实值恰好接近填充值”和“原本没有交易或没有近期记录”。正式MLP特征审计时还会确认哪些NaN需要额外missing indicator。

## 5. 为什么MLP需要标准化

假设两个特征为：

~~~text
价格变化：0.002
盘口量：100000
~~~

原始数值相差约五千万倍。虽然模型理论上可以学习很小或很大的权重来补偿，但训练刚开始时权重随机，梯度的数值尺度会严重不同：

- 某些特征主导参数更新；
- 一个统一learning rate很难同时适合所有特征；
- 优化路径可能弯曲、震荡或收敛很慢；
- 正则化对不同尺度特征的影响不公平。

标准化不是宣称所有特征同等重要，而是先把它们放到相近的数值尺度，让优化器更容易学习真正有用的权重。

## 6. 标准化公式

对某一列特征：

$$
z=\frac{x-\mu_{train}}{\sigma_{train}}
$$

其中：

- $x$：某个原始值；
- $\mu_{train}$：该列在训练集中的均值；
- $\sigma_{train}$：该列在训练集中的标准差；
- $z$：标准化后的数值。

### 最小数字例子

训练列为：

~~~text
[0, 2]
~~~

训练均值：

$$
\mu_{train}=1
$$

使用总体标准差：

$$
\sigma_{train}=1
$$

标准化后：

$$
0\rightarrow\frac{0-1}{1}=-1
$$

$$
2\rightarrow\frac{2-1}{1}=1
$$

若验证值$x=3$：

$$
z_{valid}=\frac{3-1}{1}=2
$$

验证结果2大于1完全正常。标准化不是把数值限制在$[-1,1]$；它只是减去训练均值，再除以训练标准差。

## 7. StandardScaler：学习并应用均值和标准差

### 它解决什么问题

StandardScaler对每一列学习训练均值和标准差，再应用：

$$
z=\frac{x-\mu}{\sigma}
$$

### 来源与导入

~~~python
from sklearn.preprocessing import StandardScaler
~~~

StandardScaler是scikit-learn中的类。

### 创建对象

~~~python
scaler = StandardScaler()
~~~

创建对象时还没有学习任何数据统计量。

### 训练集调用

~~~python
X_train_scaled = scaler.fit_transform(X_train_imputed)
~~~

fit_transform分成两个动作：

1. fit：从训练集每列学习均值与标准差；
2. transform：使用这些统计量转换训练集。

### 验证集调用

~~~python
X_valid_scaled = scaler.transform(X_valid_imputed)
~~~

transform不会重新计算验证均值和标准差，只使用训练阶段已经保存的统计量。

### 输入和返回值

- 输入：二维数值DataFrame或NumPy数组，shape为$(样本数,特征数)$；
- 返回：二维NumPy数组，shape不变；
- 每列分别处理，不会混合不同样本的行。

### 训练后属性

fit以后可以查看：

~~~python
print(scaler.mean_)
print(scaler.scale_)
~~~

- mean_：每列训练均值；
- scale_：每列用于缩放的尺度，通常是训练标准差。

它们是训练后属性，不加圆括号。

### 常见错误

错误：

~~~python
X_valid_scaled = scaler.fit_transform(X_valid_imputed)
~~~

这会用验证数据重新覆盖scaler中的统计量，造成泄漏和不一致。

正确：

~~~python
X_valid_scaled = scaler.transform(X_valid_imputed)
~~~

## 8. 正确处理顺序

本课采用：

~~~text
先按月份拆分train/valid
→ imputer只在train fit
→ 填充train和valid
→ scaler只在填充后的train fit
→ 标准化train和valid
→ 转换为float32 Tensor
~~~

为什么先填充再StandardScaler？因为均值、标准差和神经网络都需要有效数值；让NaN直接进入scaler会使流程语义不清，也不利于后续一致部署。

转换Tensor：

~~~python
train_features_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
)
~~~

StandardScaler通常返回float64数组，而神经网络参数通常是float32，所以这里显式指定dtype。

## 9. 当前比赛中的边界

### 标准化只作用于输入X

本课只处理输入特征。target是否缩放会影响训练loss和最后反变换，是另一个实验决定，不在本课同时引入。

### 不重新切分月份

继续使用固定对照：

~~~text
train：month 0～59
valid：month 60～70
~~~

### 不从验证集学习任何统计量

验证集可以被transform和评分，但不能更新imputer、scaler或模型权重。

### 标准化不会恢复时间信息

它只改变每列的数值尺度，不改变行数、列数、sample_id对应关系，也不会恢复聚合时丢失的逐时刻顺序。

## 10. 你的动手任务

打开exercises/lesson_05d_impute_and_standardize.py。

固定数据、路径无关代码和Tensor转换已经提供。你亲手完成：

1. imputer在训练集fit_transform；
2. 同一个imputer对验证集transform；
3. scaler在填充后的训练集fit_transform；
4. 同一个scaler对验证集transform。

运行前预测：

- 第一列的训练中位数；
- 两列的训练均值；
- 处理后的train和valid shape是否改变。

判断题：

- A. 验证集标准化值可以大于1，这不代表StandardScaler出错。
- B. imputer和scaler都应只在训练部分fit，验证部分只transform。
- C. 为了让每个验证batch都接近均值0，应该对每个验证batch重新fit scaler。

其中两个合理，一个大概率错误。完成后告诉Codex“已完成5D”。

## 11. 人话总结

~~~text
填充缺失：防止NaN污染预测、loss和梯度
保留缺失信息：用计数、比例或indicator表达“为什么缺”
标准化：让不同量纲特征处于相近数值尺度
训练集fit：学习允许使用的过去统计量
验证集transform：模拟真正未知的未来数据
~~~
