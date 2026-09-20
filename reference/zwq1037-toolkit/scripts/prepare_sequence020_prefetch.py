"""Overlap bounded CPU input preparation with GPU work; retain checkpoint format."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'scripts/exp_sequence_020_subsecond_gru_transformer_bfloat16.py').read_text(encoding='utf-8')
source = source.replace('import ast\n', 'import ast\nfrom concurrent.futures import ThreadPoolExecutor\n')
old = '''            raw = np.asarray(self.events[source][row], dtype=np.float32)
            valid = np.arange(len(raw)) < self.lengths[source][row]
            normalized = np.clip((raw - self.norm[source][0]) / self.norm[source][1], -8, 8)
            normalized[~valid] = 0
            stream = np.concatenate([normalized, valid[:, None].astype(np.float32)], axis=1)'''
new = '''            length = int(self.lengths[source][row])
            # 只读取有效事件，保持原来的固定长度和填充语义。
            # Read valid events only; retain identical fixed shapes and padding.
            raw = np.asarray(self.events[source][row, :length], dtype=np.float32)
            stream = np.zeros((self.events[source].shape[1], raw.shape[1] + 1), dtype=np.float32)
            stream[:length, :-1] = np.clip((raw - self.norm[source][0]) / self.norm[source][1], -8, 8)
            stream[:length, -1] = 1'''
assert source.count(old) == 1
source = source.replace(old, new)
prefetch = '''
def prefetched_batches(loader):
    # 只预读一批，线程共享mmap，避免Windows进程复制大缓存。
    # Prefetch one batch in a shared-memory thread, never pickle large memmaps.
    iterator = iter(loader)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(next, iterator, None)
        while True:
            batch = pending.result()
            if batch is None:
                break
            pending = executor.submit(next, iterator, None)
            yield batch

'''
source = source.replace('@torch.no_grad()\ndef predict', prefetch + '@torch.no_grad()\ndef predict')
source = source.replace('for batch_index, batch in enumerate(epoch_loader, start=skip):', 'for batch_index, batch in enumerate(prefetched_batches(epoch_loader), start=skip):')
destination = root / 'scripts/exp_sequence_020_subsecond_gru_transformer_prefetch.py'
destination.write_text(source, encoding='utf-8')
print(destination)
