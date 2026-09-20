# 第5I/J课：在真实62维特征上训练、验证并保存MLP

## 1. 本课目标：把已经学过的东西接起来

你已经做过时间划分、验证评分和选择最佳模型；5H又完成了PyTorch训练循环。因此按你的建议，原来的5I和5J合并，不再安排一节小数据验证课。

这一课不换赛题、不重建特征：把LightGBM用过的62个聚合特征交给MLP。

新增重点归为两组：

1. PyTorch如何“只考试、不学习”，并保存考试成绩最好的参数；
2. 把更新规则由SGD换成Adam，并理解它为什么还需要学习率。

数据读取、预处理、训练循环和实验文件管理已提供。你亲手补验证前向、全验证集评分和最佳模型选择，避免重抄旧代码。

本课的成功标准是完整且可信的MLP实验，不是必须超过LightGBM。

## 2. 配套阅读：只补Adam直觉，不重读验证入门

- 资料：《动手学深度学习》第二版，PyTorch中文版，`references/books/D2L_2nd_PyTorch_CN.pdf`。
- 位置：11.10“Adam算法”和11.10.1“算法”，书本488～489页，PDF阅读器506～507页。
- 时机：先听本讲义第5节的解释，再读教材，约5～10分钟。
- 深度：理解“保留梯度历史”和“按各参数的梯度规模调整更新”两个想法。
- 可跳过：前面章节的回顾清单、偏差修正推导、11.10.2从零实现和Yogi。本课不需要提前读完优化章节。
- 读后问题：① Adam替代的是backward，还是参数更新规则？② 换成Adam以后，为什么还不能保证验证分数一直上升？

验证集定义、前向时间划分和余弦公式都是已掌握的前置知识，不重复布置阅读。`eval/no_grad/save`等API由我在这里讲清楚，不要求你读官方接口文档。

## 3. 先明确这次模型看到什么

```text
已有40个全段特征 + 22个最后60秒特征
                  ↓
按月份：0～59训练，60～70验证
                  ↓
仅用训练数据学习填充值与标准化统计量
                  ↓
每样本62个数 → Linear(62,64) → ReLU → Linear(64,1)
                  ↓
训练用MSE；每轮验证用全体cosine选择最佳参数
```

`sample_id`只负责对齐，`month`只负责划分，`target`是答案，三者都不作为输入。

这里有一个包含64个单元的隐藏层，不是64层。对大小为1024的满批：

| 对象 | shape | 含义 |
|---|---|---|
| batch_features | `(1024, 62)` | 1024个样本的聚合特征 |
| 隐藏层输出 | `(1024, 64)` | 每个样本的64个学习到的组合值 |
| predictions、batch_targets | `(1024, 1)` | 每样本一个连续预测/标签 |

MLP仍然只看到聚合表，没有直接看到原始时间序列。逐时刻的模式留到CNN再学。

### 两种运行规模，不是两节课

- `SMOKE_TEST = True`：训练期固定抽20000行、验证期固定抽5000行，跑3轮，检查代码。
- `SMOKE_TEST = False`：训练1064163行、验证193474行，跑20轮，完成正式对照。

先划分，再各自抽样，绝不把验证月份抽进训练。小规模阶段的填充器和标准化器只在选中的20000个训练样本上fit；正式运行会重新用完整训练集fit。

小规模分数不与全量LightGBM比较，因为训练数据和验证样本都不同。它即使很低或是负数，也不等于正式MLP必然失败。

## 4. 验证像考试：停止学习，但仍然做计算

训练需要算出梯度，再改变参数；验证需要的是“现在这套参数究竟答得怎样”。因此验证仍然需要前向预测，只是不反向传播，也不step。

### 4.1 `model.eval()`：切换层的工作模式

这是PyTorch模型的方法，无需新import，基本写法：

```python
# 设置验证模式，但这行本身不生成预测。
# Set evaluation mode; this line alone makes no predictions.
model.eval()
```

它没有必须传入的参数，返回模型自身；一般不用接住。它对应你已用过的`model.train()`。

它影响的是会区分训练与验证行为的层，例如未来会遇到的Dropout。当前只有Linear/ReLU，两种模式的前向计算没有区别，但仍保留正确习惯。

最容易混淆的一点：`eval()`不会自动关闭梯度记录，不会自动冻结所有参数，也不会自动完成验证循环。

### 4.2 `with torch.no_grad():`：这块代码不建立反向传播所需的计算图

`torch.no_grad()`来自PyTorch，无需额外import；不传参数，返回一个上下文管理器，而不是预测或模型。

Python的`with`表示“在下面这块缩进代码执行期间，临时使用这个设置”。离开代码块后，会恢复之前的梯度记录状态。

最小例子：

```python
import torch

# x是一个允许求梯度的标量Tensor。
# x is a scalar tensor that allows gradient computation.
x = torch.tensor(2.0, requires_grad=True)
tracked_result = x * 3

# 算出的数值仍然是6，但这次不记录求导链条。
# The value is still six, but this operation does not record a gradient graph.
with torch.no_grad():
    untracked_result = x * 3

print(tracked_result.requires_grad)    # True
print(untracked_result.requires_grad)  # False
```

`requires_grad`是Tensor的布尔属性，不是方法，所以后面没有括号。两次结果都为6，差别是是否保留求导关系。

为什么验证需要它？我们不打算用验证集梯度训练模型，没必要为这段预测保存反向传播的信息，可以减少内存占用。

它也不是“锁死权重”：如果你在里面错误地调用optimizer.step，优化器仍可能使用之前留存的梯度更新参数。因此验证块里根本不应该写step。

### 两者一起用，但分工不同

- `eval()`：让层按验证方式工作。
- `no_grad()`：本段前向计算不记录反向求导图。
- 验证循环：只前向，没有`backward()`或`step()`。
- 下一轮训练：重新`model.train()`，并在no_grad块外执行原来的训练步骤。

你不需要重新实现“怎样划分验证集”；这次学的是这些PyTorch特有的开关。

## 5. Adam：把更新规则换得更灵活

你学过的普通SGD只用当前梯度。例如权重0.5、梯度2、学习率0.1，一步后是`0.5 - 0.1×2 = 0.3`。

但不同参数的梯度规模可能不同，同一个参数的梯度也可能一会儿正、一会儿负。Adam给每个参数多记两本账：

- 最近的梯度总体朝哪个方向？
- 最近的梯度数值通常有多大？

它用历史的平滑信息确定方向，再结合梯度规模调整步幅。不是每个参数都机械地减去“学习率乘当前梯度”。

用简化记号表示其更新形式：

$$
w_{new}=w_{old}-\eta\frac{\widehat m}{\sqrt{\widehat v}+\epsilon}
$$

其中：$w$是参数，$\eta$是学习率；$\widehat m$是经过初始偏差修正的梯度移动平均；$\widehat v$是经过修正的梯度平方移动平均；$\epsilon$是避免分母为零的小数。

举例：从零历史开始，第一次梯度为2，修正后$\widehat m=2$、$\widehat v=4$。若学习率0.001，忽略极小的$\epsilon$，更新量约为`0.001×2/2=0.001`，权重由0.5变为0.499。后续会混合历史信息，不能永远按第一步这样简化。

注意：这里“梯度的平方”不是LightGBM课里的二阶导数。Adam不需要你计算Hessian。

### 创建方法与你已经熟悉的SGD几乎一样

```python
# 创建Adam优化器，负责模型参数，学习率作为明确的起点。
# Create Adam for the model parameters with an explicit starting learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

来自`torch.optim`，参数是模型Parameter的可迭代对象和浮点学习率，返回Adam优化器。创建时不训练；仍然由`backward()`提供梯度，由`step()`更新参数。

本课保持默认`betas=(0.9, 0.999)`和`eps=1e-8`，不设置权重衰减。它们控制历史平滑和数值稳定；现在不扫描参数。学习率0.001只是起点，不是保证有效的最佳值。

Adam不保证比SGD泛化更好，也不保证MLP超过LightGBM。我们先搭可靠基线，不把换优化器和调网络同时变成大规模搜索。

## 6. 将分批预测拼起来，只算一次cosine

你之前已经实现过cosine。本课复用函数，不要求再抄点积和范数。

新增的问题只是：验证分批运行时，每次只得到一部分预测，怎样恢复完整顺序？

### 列表收集与`torch.cat`

```python
import torch

# 模拟两批预测：第一批2行，第二批1行。
# Simulate two prediction batches: two rows followed by one row.
prediction_batches = []
prediction_batches.append(torch.tensor([[0.1], [0.2]]))
prediction_batches.append(torch.tensor([[0.3]]))

# 沿第0维，也就是行方向，连接为3行1列。
# Concatenate along dimension zero (rows), producing three rows and one column.
all_predictions = torch.cat(prediction_batches, dim=0)
```

`append()`是Python列表方法，把一个对象加入列表末尾，返回None；不要写成`列表 = 列表.append(...)`。

`torch.cat()`是PyTorch函数，输入Tensor列表和连接维度`dim`，返回一个新的Tensor。上例从`(2,1)+(1,1)`得到`(3,1)`；除连接维度外，其他维度必须一致。空列表不能cat。

### 为什么先`.cpu()`，再`.numpy().reshape(-1)`

这些转换已在固定代码中提供，只需要看懂顺序：

1. `batch_predictions.cpu()`：Tensor方法，无参数，返回位于CPU的Tensor，shape不变。只将预测搬回CPU，不把所有数据搬上GPU。
2. `validation_tensor.numpy()`：CPU Tensor方法，返回NumPy数组，shape保持`(N,1)`。本课预测已经由no_grad产生，不带求导要求；不能直接把普通CUDA Tensor转NumPy。
3. `prediction_array.reshape(-1)`：NumPy数组方法，返回一维数组`(N,)`，`-1`表示根据元素总数推断长度。这里不是倒序索引；例：`[[0.1],[0.2]]`变为`[0.1,0.2]`。

练习里的紧凑写法：

```python
validation_predictions = validation_tensor.numpy().reshape(-1)
```

等价于：

```python
# 分两步转成固定余弦函数使用的一维数组。
# Convert in two steps to the one-dimensional array expected by the fixed scorer.
prediction_array = validation_tensor.numpy()
validation_predictions = prediction_array.reshape(-1)
```

配套函数`cosine_similarity_score(y_true, y_pred)`来自同目录的`lesson_05ij_support.py`，已import。输入是两个同长度一维数组`(N,)`，返回Python float；零向量返回0，形状或非有限数值错误会被检查拦住。

脱离比赛的用法：`cosine_similarity_score([1, 0], [1, 0])`得到1。在本课，参数应换成全验证集标签和预测，不是最后一批。

例如`y=[1,1]、p=[1,10]`：若各自拆成单样本批，每批cosine都是1，平均仍为1；但完整cosine为$11/(\sqrt{2}\sqrt{101})\approx0.774$。所以不能平均每批cosine来替代全体验证分数。

验证DataLoader固定`shuffle=False`，收集预测时与外部标签数组保持相同顺序；千万不能只把其中一方打乱。

## 7. 保存最佳模型：存的是那一轮的参数，不是最后一轮

假设验证cosine依次为0.03、0.06、0.04，应留下第二轮。第三轮训练loss即使更低，也不改变这一选择规则。

### `model.state_dict()`：拿到参数清单

这是模型的方法，无必填参数，返回一个字典式映射：名字是键，参数和持久buffer Tensor是值。

在本课的Sequential里，例如`"0.weight"`代表第一层权重，shape为`(64,62)`。这份清单不是完整Python模型结构，也不包括填充器或标准化器。

### `torch.save(obj, path)`：立即写到文件

来自PyTorch，输入待保存对象与路径（字符串或Path），写入文件，返回None。

最小例子（与本课答案的变量名不同）：

```python
# 将一个小Tensor字典写入文件；相同文件名再次保存会覆盖该文件。
# Save a small tensor dictionary; saving to the same path overwrites that file.
example_state = {"value": torch.tensor([1.0, 2.0])}
torch.save(example_state, "example_state.pt")
```

本课要把模型参数清单保存到已提供的`best_model_path`，并且只在分数刷新最佳值时保存。

不能只写`best_state = model.state_dict()`，等训练结束才保存它：其中Tensor可能仍引用模型当前存储，会跟着后续训练改变。这里选择每次改善就立即写磁盘，避免这个陷阱。

初始`best_cosine = -float("inf")`：`float("inf")`是Python内置转换得到正无穷，前面的负号变为负无穷。这样第一次有效分数即使是负数，也能被保存；不能默认初始最佳值为0。

### 保存的不止权重，但不要求你重写文件管理

每次运行创建独立目录`outputs/lesson_05ij/smoke_随机后缀/`或`full_随机后缀/`，不会覆盖LightGBM或前一次实验。

| 文件 | 保存内容 |
|---|---|
| best_model.pt | 验证cosine最好的那轮参数 |
| preprocessing.joblib | 训练期拟合的填充器、标准化器与有序特征名 |
| config.json | 数据规模、月份、网络、学习率等设置 |
| best_valid_predictions.feather | 最佳轮的sample_id、month、target、prediction |
| best_result.json | 最佳轮数与cosine |
| history.json | 每轮训练MSE、验证cosine和耗时 |

主脚本保存权重后，配套函数立即保存同一轮的预测与分数，保证三者对应。不额外重跑LightGBM。

这是“可恢复用于预测”的保存方案；本课没有保存Adam状态，不能声称能完全接着中断点继续训练。5K会检查重新加载后的预测一致性。只加载自己生成或可信来源的模型/预处理文件。

## 8. 已提供代码的阅读地图

主文件：`exercises/lesson_05ij_real_mlp.py`，这是你要补的文件。

配套文件：`exercises/lesson_05ij_support.py`，可以略读注释，不用修改或手写。它不是新的第三方库，同目录import会找到它。

### 四个提供的辅助函数

- `prepare_data(project_dir, smoke_test)`：输入项目Path和布尔开关，返回字典。`prepared["train_features"]`等四个训练/验证特征与标签是CPU float32 Tensor；`valid_labels`是按相同顺序排列的DataFrame；其余键保存特征名及拟合好的预处理器。这里的`[]`是按字典键取对象，不是按行切片。
- `create_run_directory(project_dir, smoke_test, prepared, num_epochs, learning_rate)`：输入路径、开关、准备结果和配置，创建独立目录并保存预处理与配置，返回Path。输出路径打印在控制台。
- `save_best_outputs(run_dir, valid_labels, predictions, epoch_number, score)`：输入目录、DataFrame、一维预测、轮数和浮点分数，保存最佳轮配套结果，返回None。
- `save_history(run_dir, history)`：输入目录和由字典构成的列表，保存逐轮日志，返回None。

这些函数的实际调用已全部写好；不需要为了练习重新学习一套文件管理。不要把`prepare_data()`放进epoch循环，否则每轮都会重复读数据和fit。

### 为什么这次不会一下子加载整个market

辅助文件只读取现成的两个聚合表。它先读取ID建立位置映射，然后一次读取一个特征列，填入预分配数组。训练/验证数组合计约312MB（十进制），预处理仍需要额外临时内存；关闭不必要的软件会更稳妥。

- `pd.Index(...).get_indexer(...)`把目标ID映射到源表行位置，不假设“第几行就是哪个sample_id”。找不到时返回-1，代码会立刻检查，不能悄悄拿最后一行。
- `np.empty(shape, dtype=np.float32)`预留数组空间，起初内容未初始化，所以每一列都必须填完才能训练。这部分固定代码已检查。
- `SimpleImputer(..., keep_empty_features=True, copy=False)`保留训练期全缺失的列并以0处理，使维度保持62；copy=False允许复用内存，不保证所有情况下都不复制。其余列仍使用训练中位数。
- `StandardScaler(copy=False)`只在训练部分fit，验证部分仅transform，尽可能原地缩放。
- `torch.from_numpy(array)`将NumPy数组转成共享存储的CPU Tensor，shape不变；例：float32的`(2,3)`数组得到float32的`(2,3)` Tensor。准备后不再改原数组，避免Tensor一起改变。
- `num_workers=0`让DataLoader在当前进程读取数据，不启动额外Windows加载进程；不是不用GPU。
- `torch.isfinite(loss).item()`检查标量loss是否是正常有限数；如果不是，`raise ValueError(...)`会主动停止。`assert`检查不满足时也会报错停止，而不是继续输出不可信分数。

这些是提供的准备与防错设施，不另布置编码题。遇到看不懂的具体行可以直接问我。

## 9. 你的任务：六个小位置，不重写大流程

按顺序补主脚本TODO：

1. 设置验证模式。
2. 给验证代码块关闭梯度记录，保留with和缩进。
3. 用当前批特征得到预测。
4. 对全部验证标签和预测计算一次cosine。
5. 判断本轮分数是否严格优于历史最佳。
6. 立即保存当前模型的参数清单。

先保留`SMOKE_TEST=True`运行。省略号必须先填好；模型结构、优化器和数据准备不要随手改。固定代码中的MSE依然用来训练，cosine只负责评分和选择模型。

写完后告诉我“5I/J试跑完成”，我会先检查代码、运行结果和保存文件。通过后，你只把开关改成False，再运行完整实验，不另开一节课。

### 验收标准

- 试跑shape为`(20000,62)`与`(5000,62)`；全量为`(1064163,62)`与`(193474,62)`。
- 严格使用训练月份0～59、验证60～70；预处理fit不接触验证样本。
- 验证块没有backward/step，预测不保留求导图，各批预测与标签完整对齐。
- 最佳分数等于history里的最高验证cosine；对应的参数、预测、轮数文件一致，不能只保存最后一轮。
- 所有损失和分数有限；若报错，先保留报错现场让我看，不通过删除验证检查来“跑通”。
- 完成主脚本末尾两个简短复盘点。它们检查PyTorch特有开关和验证底线，不重复考验证定义。

### 正式结果怎么解释

全量才与EXP-002的cosine `0.1038700392`做对照。两者用相同样本与同一组62个原始聚合特征，但MLP新增必要的填充与标准化，所以比较的是两套建模流程，不是证明“只换Adam”的因果效果。

若MLP较弱，先检查正确性，再如实记录；不要靠把验证集加入训练或不断试很多设置追这个验证分数。单次固定时间划分也不能证明长期稳健性。

## 10. 人话总结

训练时改答案策略，验证时只检查当前策略，成绩最好时立即存档。Adam改变怎么更新参数，不改变验证底线。

你完成核心代码并做试跑；我检查后再让你切全量。整个5I/J通过后进入5K：恢复模型、复核预测、总结与LightGBM的差异，然后转向CNN，不再拆回两节重复课。

## 教师接口核对来源（非额外阅读任务）

- [PyTorch 2.9 no_grad](https://docs.pytorch.org/docs/2.9/generated/torch.no_grad.html)
- [PyTorch 2.9 Module](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html)
- [PyTorch 2.9 Adam](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Adam.html)
- [PyTorch保存与加载模型教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
