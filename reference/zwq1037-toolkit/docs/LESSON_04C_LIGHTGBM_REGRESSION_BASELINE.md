# 第 4 课 C：第一次 LightGBM 时间验证训练

## 本课所处阶段

阶段 4：建立 LightGBM 强基线。

现在终于开始正式模型训练。数据已经满足：

```text
X_train：月份 0～59，(1,064,163, 40)
X_valid：月份 60～70，(193,474, 40)
y：连续 target
验证指标：Cosine Similarity
```

你在上一项目已经掌握 boosting、LightGBM、early stopping 和 `best_iteration_`。本课不重讲这些内容，只处理当前比赛与上一场分类赛不同的地方。

## 本课目标

完成后，你应该能够：

1. 区分 `LGBMClassifier` 与 `LGBMRegressor`。
2. 区分回归中的 `predict()` 与分类中的 `predict_proba()`。
3. 解释训练目标（objective）和早停评价指标（eval metric）为什么可以不同。
4. 让 LightGBM 每轮计算验证集余弦相似度，并根据“越大越好”early stop。
5. 得到、保存并记录第一个可信的月份前向验证分数。

本课不做这些事情：

- 不调很多组参数；
- 不加入第二档盘口、order 或 transaction 原始事件流；
- 不把 month 放入 `X`；
- 不使用随机 K-fold；
- 不融合模型；
- 不根据一次验证结果就断言某个特征一定有效。

## 配套阅读

### 本课不安排新的外部阅读

原因是：

- LightGBM、boosting 和 early stopping 已经在上一项目实际训练过。
- `LGBMRegressor`、`predict()`、`eval_X`、`eval_y` 和自定义评价函数属于接口变化，由本讲义直接讲解。
- 余弦相似度已在第 2A 课学过，本课是把它接入训练过程。

完成基线后，下一次真正需要补充教材时，再针对回归误差、正则化或时间验证选择准确章节。

## 1. 上一项目是分类，这次是回归

上一项目的答案是：

```text
0：不成瘾
1：成瘾
```

所以使用：

```python
LGBMClassifier
```

并预测类别 1 的概率：

```python
probabilities = model.predict_proba(X_valid)[:, 1]
```

当前 target 是连续数值，例如：

```text
0.005409
0.000610
-0.002765
```

这不是类别编号，因此使用回归器：

```python
LGBMRegressor
```

它直接输出一个连续预测值：

```python
predictions = model.predict(X_valid)
```

如果验证集有 193,474 行：

```text
predictions.shape == (193474,)
```

没有“第 0 类概率”和“第 1 类概率”，所以不再写 `[:, 1]`。

## 2. `LGBMRegressor`

### 它解决什么问题

使用 LightGBM 的梯度提升树预测连续数值。

### 来源与导入

它来自第三方库 `lightgbm`：

```python
from lightgbm import LGBMRegressor
```

### 基本语法

```python
model = LGBMRegressor(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    reg_lambda=1.0,
    metric="None",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)
```

返回：尚未训练的 `LGBMRegressor` 对象。

### 本课参数为什么这样设？

- `objective="regression"`：使用回归损失学习连续 target。
- `n_estimators=5000`：最大轮数，不代表一定训练满。
- `learning_rate=0.03`：每棵树迈较小一步。
- `num_leaves=31`：使用熟悉的普通起点，不在本课调参。
- `min_child_samples=100`：数据超过百万行，避免叶子只依赖极少样本。
- `reg_lambda=1.0`：加入基础 L2 正则。
- `metric="None"`：关闭默认的验证 L2 显示，让early stopping只观察本比赛的余弦指标。
- `random_state=42`：固定随机状态。
- `n_jobs=-1`：使用可用CPU核心。
- `verbosity=-1`：关闭大量非必要提示。

这些参数只是V1基线，不代表最佳参数。

### 输入和输出

训练：

```python
model.fit(X_train, y_train, ...)
```

- `X_train.shape == (1064163, 40)`。
- `y_train.shape == (1064163,)`。
- `fit()` 后模型产生 `best_iteration_`。

预测：

```python
validation_predictions = model.predict(
    X_valid,
    num_iteration=model.best_iteration_,
)
```

- 输入 shape：`(193474, 40)`。
- 返回 NumPy 一维数组，shape：`(193474,)`。
- `num_iteration` 明确只使用最佳轮数的树。

### 常见错误

不要继续使用 `predict_proba()`。回归器输出连续值，不输出类别概率矩阵。

## 3. 训练目标和比赛指标不是一回事

模型每棵树需要知道怎样修正每一个样本。普通回归目标可以使用平方误差：

$$
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-p_i)^2
$$

它能为每个样本提供可计算的误差和梯度。

比赛最终评价整条预测向量的方向：

$$
\operatorname{cosine}(y,p)
=
\frac{y\cdot p}
{\lVert y\rVert_2\lVert p\rVert_2}
$$

所以本课采用：

```text
objective="regression"
→ 决定树怎样学习

eval_metric=余弦相似度
→ 决定哪一轮验证表现最好、何时停止
```

模型仍然根据回归损失长树，但我们用比赛指标选择最佳轮数和报告最终结果。

## 4. LightGBM 自定义评价函数

第 2A 课已经有普通余弦评分函数：

```python
score = cosine_similarity_score(y_true, y_pred)
```

LightGBM 需要评价函数返回三个值：

```python
def lightgbm_cosine_metric(y_true, y_pred):
    score = cosine_similarity_score(y_true, y_pred)
    return "cosine", score, True
```

返回值是一个 tuple：

```text
"cosine"：指标名称
score：本轮数值
True：越大越好
```

如果第三项错误地写成 `False`，early stopping 会把更低的余弦分数误认为更好。

这个函数的代码已经提供，不要求你重写。

## 5. `eval_X` 和 `eval_y`

### 它解决什么问题

告诉 LightGBM 训练过程中要在哪份特征和标签上观察评价指标。

### 基本语法

当前环境是 LightGBM 4.7.0。这个版本仍能识别旧式 `eval_set=[(X_valid, y_valid)]`，但会给出弃用警告，因此本课使用当前推荐接口：

```python
eval_X=X_valid,
eval_y=y_valid,
```

- `eval_X`：二维验证特征，shape 为 `(193474, 40)`。
- `eval_y`：一维验证标签，shape 为 `(193474,)`。
- 两者行数必须相同，并且顺序一一对应。

这不会让验证样本参与树的切分学习：

```text
X_train, y_train → 学习树
X_valid, y_valid → 观察每轮表现与决定停止
```

### 常见错误

不能把 Kaggle 测试集作为 `eval_X`、`eval_y` 决定停止，因为测试集没有真实 target。

## 6. 两个训练回调

你已经学过 `early_stopping()`。本课使用：

```python
stopping_callback = early_stopping(
    stopping_rounds=200,
    first_metric_only=True,
    verbose=True,
)
```

含义：余弦分数连续200轮没有创造新最佳值就停止。

为了避免训练数分钟却没有任何进度输出，固定代码还导入：

```python
from lightgbm import log_evaluation
```

并创建：

```python
logging_callback = log_evaluation(period=100)
```

它返回一个日志回调对象，每100轮打印一次验证指标。它不改变模型学习结果，只负责显示进度。

传入 `fit()`：

```python
callbacks=[stopping_callback, logging_callback]
```

## 7. `fit()` 的完整连接方式

```python
model.fit(
    X_train,
    y_train,
    eval_X=X_valid,
    eval_y=y_valid,
    eval_names=["forward_valid"],
    eval_metric=lightgbm_cosine_metric,
    callbacks=[stopping_callback, logging_callback],
)
```

新参数：

- `eval_X=X_valid`、`eval_y=y_valid`：当前 LightGBM 4.7 接口中的验证特征与标签。
- `eval_names=["forward_valid"]`：给日志中的验证集起名。
- `eval_metric=lightgbm_cosine_metric`：每轮调用我们的余弦评价函数。
- `callbacks`：同时使用早停观察员和日志记录员。

返回值仍是训练好的模型对象；本课直接使用已经保存模型的变量，不重新接收返回值。

## 8. 内存为什么要主动释放？

完整特征、label、合并表和训练/验证副本会同时占用内存。固定代码在切分完成后执行：

```python
del feature_data
del label_data
del model_data
gc.collect()
```

`del variable_name` 是Python语句，删除当前变量名对对象的引用；`gc.collect()` 请求垃圾回收器尽快回收已经无法访问的对象。

它不会删除磁盘文件，也不会删除 `X_train` 等仍在使用的对象。这里是为了给LightGBM训练留出更多内存。

## 9. 保存哪些结果？

训练成功后，固定代码保存：

```text
outputs/models/lightgbm_market_v1.txt
outputs/predictions/lightgbm_market_v1_valid.feather
```

验证预测文件包含：

```text
sample_id
month
target
prediction
```

这样后面可以做误差分析、融合和复查，不必为了查看预测重新训练模型。

当前项目路径包含中文。LightGBM 4.7 的底层 C++ `save_model()` 在此 Windows 环境中不能可靠写入该路径，因此固定代码使用：

```text
model_to_string() 生成模型文本
→ Python Path.write_text(..., encoding="utf-8") 写入文件
```

这只改变保存途径，不改变模型内容、最佳轮数或预测结果。

## 10. 运行前预测

运行前请在代码文件开头附近用中文写三句预测，不需要中英双语：

1. 你认为V1聚合特征能得到正的验证余弦分数吗？
2. 你认为训练会在5000轮之前early stop吗？
3. 你认为 `mid_price_1_change` 会比平均盘口量更重要吗？为什么？

预测不要求正确，目的是让结果修正直觉。

## 11. 你的动手任务

练习文件：

```text
scripts/lesson_04c_train_lightgbm_baseline.py
```

请完成四组TODO：

1. 根据第2节参数创建 `LGBMRegressor`。
2. 创建 `stopping_rounds=200` 的early-stopping回调。
3. 按第7节接口调用 `model.fit()`。
4. 用最佳轮数 `predict()`，再调用余弦评分函数。

数据读取、merge、时间划分、余弦函数、日志回调、内存释放和结果保存已经提供。

## 验收标准

- 使用 `LGBMRegressor`，不是分类器。
- `X_train` 和 `X_valid` 分别来自月份0～59和60～70。
- `eval_X`、`eval_y` 只使用验证集，不接触Kaggle测试集。
- early stopping监控余弦相似度，并把它视为越大越好。
- 预测 shape 为 `(193474,)`。
- 打印有限的最佳轮数、训练秒数和验证余弦分数。
- 保存最佳模型与验证预测。
- 能解释objective、eval metric和最终比赛score三者的关系。

## 运行后回答

1. `LGBMRegressor.predict()` 为什么不需要 `[:, 1]`？
2. 本课模型根据什么损失学习树，又根据什么指标选择最佳轮数？
3. 本地分数是对哪些月份的未来模拟？它是否等于Kaggle最终测试分数？

## 本课人话总结

数据工程终于接到了模型上：过去60个月教模型，后面11个月当一次真正的未来考试。LightGBM仍用普通回归误差学习怎样修正预测，但每轮都接受比赛余弦指标的检查。这个分数不一定很高，却是后续所有特征、参数和深度模型必须公平超越的第一条可信基线。
