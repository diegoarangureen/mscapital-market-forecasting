# 第 3 课 A：不读取整表，先查看 market 的结构

## 本课所处阶段

阶段 3：安全读取并对齐 market 小样本。

完整的训练 `market.feather` 为 `4,396,491,768` 字节，现已下载到本地。本课在 PyCharm 中以内存映射方式查看文件结构和第一个安全的存储批次，不把整张表转换成 pandas DataFrame。

## 本课目标

完成后，你应该能够：

1. 区分文件大小、内存占用和内存映射。
2. 使用 Arrow IPC reader 查看 Feather 文件的 `schema`。
3. 解释 RecordBatch 是存储批次，不等于一个预测样本，也不等于训练 batch。
4. 解释为什么本课不调用 `read_all()` 或 `pd.read_feather()`。

本课不做这些事情：

- 不训练模型；
- 不处理 order 或 transaction；
- 不一次读取全部 market 数据；
- 不猜测 market 的列名，必须以实际输出为准；
- 不在本课完成 label 对齐，那是第 3B 课。

## 配套阅读

### 课前浏览：Apache Arrow 官方 Feather 说明

- 资料：[Apache Arrow - Feather File Format](https://arrow.apache.org/docs/python/feather.html)
- 版本：在线文档；本机当前是 PyArrow 25.0.1。
- 范围：只读开头对 Feather V1/V2 的说明，以及 `read_feather()` 与 `read_table()` 返回对象的区别。
- 深度：理解直觉即可，用时约 5～10 分钟。
- 暂时跳过：压缩参数、Feather V1 兼容和写入细节。

### 课后查阅：Arrow IPC reader

- 资料：[RecordBatchFileReader](https://arrow.apache.org/docs/python/generated/pyarrow.ipc.RecordBatchFileReader.html)
- 范围：只看 `schema`、`num_record_batches` 和 `get_batch(i)` 三项。
- 深度：知道它们分别是两个属性和一个方法，不需要阅读底层 IPC 协议。
- 暂时跳过：custom metadata、memory pool、writer 和 stream reader。

阅读后回答：

1. `read_feather()` 返回 pandas DataFrame，`ipc.open_file()` 返回什么？
2. 为什么能查看 `schema` 不代表整张 market 表已经进入内存？
3. `get_batch(0)` 中的 `0` 为什么是第一个批次？

### 本课暂不安排的资料

- ISLP：本课是大文件读取，不是统计学习理论。
- 《动手学深度学习》：还没有进入 Tensor 和神经网络。
- Quant Wiki：第 1A 课已经读过限价单和限价单簿。本课没有出现必须补充的新市场术语，因此不重复安排；以后只在出现新的量化概念时指定新页面。

## 1. 为什么不能直接 `pd.read_feather()`？

文件在磁盘上占 4.40 GB，不代表恢复成 DataFrame 后仍然只占 4.40 GB。解压、数据类型、索引和 pandas 对象都可能增加内存占用。

直接执行：

```python
market_data = pd.read_feather(market_path)
```

表达的是“把整张表恢复为 pandas DataFrame”。这一步目前没有必要，也可能造成 Notebook 内存压力。

我们真正需要先回答的是：

```text
有哪些列？
每列是什么类型？
文件内部被分成多少个存储批次？
第一个批次大约多大？
```

## 2. 内存映射是什么？

普通读取可以想成：

```text
磁盘文件 → 一次搬进内存 → 程序使用
```

内存映射（memory mapping）更像给文件建立一张“地址地图”：

```text
磁盘文件 ↔ 虚拟内存地址
                 ↓
          访问哪部分，再读取哪部分
```

它不等于“4.40 GB 完全不占资源”，但可以避免一开始就主动把整个文件复制成一个 pandas DataFrame。

## 3. `pa.memory_map()`

### 它解决什么问题

以只读内存映射方式打开本地文件，后续 Arrow reader 可以按需要访问文件内容。

### 来源与导入

它来自 `pyarrow`：

```python
import pyarrow as pa
```

### 基本语法

```python
source = pa.memory_map(file_path, "r")
```

### 输入和返回值

- `file_path`：字符串路径。
- `"r"`：read，只读模式。
- 返回：`pyarrow.MemoryMappedFile`，它是文件访问对象，不是 DataFrame。

### 最小例子

```python
source = pa.memory_map("students.feather", "r")
```

这只是打开文件，还没有生成学生数据的 pandas DataFrame。

### 当前作用与常见错误

当前用于打开项目 `data/raw/train/market.feather`。路径必须转换为字符串；不要把模式写成 `"w"`，因为 `w` 表示写入。

## 4. `with ... as ...:`

### 它解决什么问题

文件使用结束后应该关闭。`with` 是 Python 内置的上下文管理语法，可以在缩进代码块结束时自动释放文件资源。

### 基本语法

```python
with pa.memory_map(str(market_path), "r") as source:
    # 这里可以使用 source。
    # Use source inside this indented block.
    ...
```

- `source` 是我们自己起的变量名。
- 冒号 `:` 表示下面开始一个代码块。
- 块内代码必须缩进，通常是 4 个空格。
- 离开缩进块后，文件自动关闭。

### 最小例子

```python
with open("note.txt", "r") as text_file:
    text = text_file.read()
```

这里 `open()` 是 Python 内置函数；例子只用于理解 `with`，本课实际使用 `pa.memory_map()`。

### 常见错误

忘记缩进会导致语法或作用域问题。不要在 `with` 块结束并关闭文件后，继续要求 reader 读取新的 batch。

## 5. `ipc.open_file()`

### 它解决什么问题

根据 Arrow 文件末尾记录的目录信息创建 reader，使我们可以查看 schema，并按编号访问 RecordBatch。

Feather V2 内部使用 Arrow IPC file format，因此可以这样打开。

### 来源与导入

```python
import pyarrow.ipc as ipc
```

### 基本语法

```python
reader = ipc.open_file(source)
```

### 输入和返回值

- 输入：可读取的 Arrow 文件对象，本课是 `MemoryMappedFile`。
- 返回：`RecordBatchFileReader`。
- reader 不是 DataFrame，也不是完整数据表。

### 最小例子

```python
reader = ipc.open_file(source)
```

创建 reader 后，可以查看元数据；不要在本课调用 `reader.read_all()`，因为它会读取所有 RecordBatch。

## 6. 属性与方法

reader 提供两个本课需要的属性：

```python
market_schema = reader.schema
batch_count = reader.num_record_batches
```

属性描述对象当前已有的信息，不需要圆括号：

- `schema`：所有列名和 Arrow 数据类型。
- `num_record_batches`：文件内部有多少个存储批次，返回整数。

读取指定批次使用方法：

```python
first_batch = reader.get_batch(0)
```

`get_batch()` 有圆括号，因为它是方法；参数 `0` 是批次索引。Python 从 0 开始编号，所以 `0` 表示第一个 RecordBatch。

返回的 `first_batch` 是 `pyarrow.RecordBatch`，常用属性包括：

```python
first_batch.shape         # (行数, 列数)
first_batch.nbytes        # 该批次数据大约占多少字节
first_batch.column_names  # 列名列表
```

这些都是属性，不加圆括号。

## 7. RecordBatch 不等于训练 batch

RecordBatch 是文件存储时的一块数据：

```text
一个 4.40 GB Arrow 文件
├── RecordBatch 0
├── RecordBatch 1
├── RecordBatch 2
└── ...
```

它不保证恰好对应：

- 一个 `sample_id`；
- 一个月；
- 神经网络的一次训练 batch。

神经网络训练 batch 是我们以后用 DataLoader 主动组合的样本；两者只是在中文里都叫“批次”，用途不同。

## 8. 在 PyCharm 中运行

1. 在 PyCharm 打开本课练习文件。
2. 确认解释器仍是 `D:\anaconda\envs\pytorch\python.exe`。
3. 只完成四个 TODO，不要加入 `read_all()` 或 `pd.read_feather(market_path)`。
4. 直接运行脚本；路径代码会自动找到项目中的训练 market 文件。
5. 完成后告诉我，我会读取文件并运行验收。

练习文件：

```text
scripts/lesson_03a_inspect_market.py
```

## 9. 你的动手任务

请亲手完成：

1. 使用 `ipc.open_file(source)` 创建 reader。
2. 读取 reader 的 `schema` 属性。
3. 读取 reader 的 `num_record_batches` 属性。
4. 当文件有多个 RecordBatch 时，使用 `get_batch(0)` 取得第一个批次。

路径检查、文件大小、条件判断和结果打印已经提供，不需要机械重写。

## 验收标准

- `market path exists = True`。
- 能打印 market 的真实 schema 和 RecordBatch 数量。
- 如果批次数量大于 1，能打印第一个批次的类型、shape、列名和 `nbytes`。
- 没有调用 `read_all()` 或完整读取到 pandas。
- 能解释 schema、RecordBatch 和预测样本三者的区别。

## 本课人话总结

面对一个几 GB 的陌生文件，第一步不是把它全部搬进内存，而是先读目录、看列结构，再决定真正需要哪一小块。大数据处理的第一项能力不是“读得快”，而是“知道什么暂时不该读”。
