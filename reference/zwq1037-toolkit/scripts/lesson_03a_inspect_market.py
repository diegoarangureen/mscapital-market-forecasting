from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


# 找到当前比赛项目目录。
# Locate the current competition project directory.
project_dir = Path(__file__).resolve().parents[1]

# 指向保持不变的训练 market 原始文件。
# Point to the immutable raw training market file.
market_path = project_dir / "data" / "raw" / "train" / "market.feather"

# 固定的路径与文件大小检查，不需要修改。
# Fixed path and file-size checks; no changes are needed.
print(f"market path exists = {market_path.exists()}")
print(f"market file size = {market_path.stat().st_size / (1024 ** 3):.2f} GB")
print()

# 以内存映射的只读模式打开文件；离开缩进块后自动关闭。
# Open the file as a read-only memory map; close it after leaving the block.
with pa.memory_map(str(market_path), "r") as source:
    # TODO 1：使用 ipc.open_file(source) 创建 Arrow 文件 reader。
    # TODO 1: Create an Arrow file reader with ipc.open_file(source).
    reader=ipc.open_file(source)

    # TODO 2：读取 reader 的 schema 属性。
    # TODO 2: Read the schema attribute of reader.
    market_schema=reader.schema

    # TODO 3：读取 reader 的 num_record_batches 属性。
    # TODO 3: Read the num_record_batches attribute of reader.
    batch_count=reader.num_record_batches

    # 以下代码打印文件目录信息，不需要修改。
    # The following code prints file metadata and requires no changes.
    print("market schema:")
    print(market_schema)
    print()
    print(f"record batch count = {batch_count}")
    print()

    # 如果整个文件只有一个巨大批次，本课主动停止，不冒险读取它。
    # If the file has only one huge batch, stop here instead of reading it.
    if batch_count > 1:
        # TODO 4：使用 get_batch(0) 读取第一个 RecordBatch。
        # TODO 4: Read the first RecordBatch with get_batch(0).

        first_batch=reader.get_batch(0)
        # 以下代码只观察第一个存储批次，不需要修改。
        # The following code only inspects the first storage batch.
        print(f"first batch type = {type(first_batch)}")
        print(f"first batch shape = {first_batch.shape}")
        print(f"first batch nbytes = {first_batch.nbytes}")
        print(f"first batch columns = {first_batch.column_names}")
    else:
        print("Only one RecordBatch was found; do not read it in this lesson.")
