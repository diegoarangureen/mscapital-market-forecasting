# 第 0 课：依赖、Feather 与正确的 Python 环境

## 配套阅读

本课不要求阅读机器学习或深度学习教材。依赖安装属于工程准备，直接理解本讲义并完成验证即可。

## 本课目标

完成后，你应该能够解释：

1. 什么是项目依赖（dependency）。
2. 为什么 `.feather` 文件需要 `pyarrow`。
3. `pyarrow`、`polars` 和 `psutil` 在本项目中分别做什么。
4. 为什么安装时使用 `python -m pip`，以及怎样保证包进入正确的 Conda 环境。

本课不训练模型，也不下载比赛数据。

## 1. 我们遇到了什么麻烦？

上一场比赛的 `train.csv` 只有几十 MB，pandas 可以直接读入。新比赛约有 10 GB 原始数据，最大的表包含上亿行。如果一次全部装入约 16 GB 内存，程序很容易变慢或直接内存不足。

比赛文件还是 `.feather`，不是 `.csv`。Feather 是一种二进制表格格式：它保存了列的数据类型，读取通常比逐字解析 CSV 更快。

可以把两者想象成：

```text
CSV：一张全用文字写成的表，读取时要重新判断每个值是什么类型
Feather：已经分类装箱的表格，读取程序知道每列如何还原
```

## 2. 什么是依赖？

我们的代码自己不会实现所有功能，而会调用别人写好的库。项目运行所需要的这些外部库，叫做依赖（dependency）。

例如：

```python
import pandas
```

`pandas` 不是 Python 内置功能。环境里没有安装它时，导入就会出现：

```text
ModuleNotFoundError: No module named 'pandas'
```

`requirements.txt` 是依赖清单，但把名字写进去不会自动完成安装。它更像购物清单，`pip` 才是根据清单取回并安装包的工具。

## 3. 三个新依赖分别做什么？

### `pyarrow`

- 来源：Apache Arrow 项目的 Python 库，需要安装。
- 当前作用：支持 pandas 读取 Feather 文件。
- 基本导入：`import pyarrow`
- 返回值：`import` 不返回数据，而是把模块对象绑定到名字 `pyarrow`。
- 注意：没有它时，`pandas.read_feather()` 通常无法工作。

### `polars`

- 来源：Polars 数据处理库，需要安装。
- 当前作用：后续尝试只选择需要的列、过滤月份，并减少大数据处理的内存压力。
- 常见导入：`import polars as pl`
- `as pl` 的意思：给模块起一个短别名，之后使用 `pl` 代表 `polars`。
- 注意：Polars 的接口与 pandas 相似但不完全相同，不能机械替换所有代码。

### `psutil`

- 来源：第三方系统监控库，需要安装。
- 当前作用：让检查脚本显示当前内存使用情况。
- 基本导入：`import psutil`
- 注意：它不是模型所必需的，只是帮助我们避免把电脑内存用完。

## 4. 为什么使用 `python -m pip`？

我们要运行：

```powershell
D:\anaconda\envs\pytorch\python.exe -m pip install pyarrow polars psutil
```

拆开理解：

```text
D:\anaconda\envs\pytorch\python.exe
```

明确指定已有的 `pytorch` 环境解释器。

```text
-m pip
```

`-m` 表示让这个 Python 运行名为 `pip` 的模块。这样安装位置会跟前面的 Python 解释器一致，避免包被装到另一个 Python 3.13 环境。

```text
install pyarrow polars psutil
```

让 pip 安装后面的三个包。

最常见错误是只运行 `pip install ...`，结果命令中的 `pip` 属于另一个 Python，PyCharm 使用的环境仍然找不到包。

## 5. 最小例子：模块、别名和属性

```python
# 导入 Python 内置的数学模块。
# Import Python's built-in math module.
import math

# 通过“模块名.成员名”读取模块中的 pi。
# Read pi from the module with "module_name.member_name".
circle_ratio = math.pi

print(circle_ratio)
```

这里：

- `math` 是模块对象。
- `.` 表示进入这个对象，查找它的成员。
- `pi` 是模块提供的属性，不需要加 `()`。
- 如果写成 `math.pi()`，会因为把数字当函数调用而报错。

## 6. 你的动手任务

### 任务 A：安装依赖

在 PyCharm 下方的 Terminal 中复制并运行：

```powershell
D:\anaconda\envs\pytorch\python.exe -m pip install pyarrow polars psutil
```

安装属于环境准备，直接复制命令即可，没有机械手写的学习价值。

### 任务 B：由你写验证脚本

新建文件：

```text
projects/mscapital_market_forecasting/scripts/lesson_00_dependency_check.py
```

请你亲手完成以下内容：

1. 导入 `sys`、`pyarrow`、`polars` 和 `psutil`。
2. 打印 `sys.executable`，确认正在使用哪个 Python。
3. 分别打印三个第三方库的 `__version__`。

接口提示：

```text
模块对象.__version__
```

`__version__` 是库提供的版本属性，通常是字符串，不要在后面加 `()`。

## 验收标准

- 脚本没有 `ModuleNotFoundError`。
- `sys.executable` 显示 `D:\anaconda\envs\pytorch\python.exe`。
- 屏幕上能看到 `pyarrow`、`polars`、`psutil` 三个版本号。

完成后告诉 Codex“第 0 课完成”，Codex 会先读取并运行你的文件，不会直接修改。
