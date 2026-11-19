"""Batch assembly and bounded, ordered CPU prefetch (framework-independent)."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque

def batch_layout(effective_batch, batch_size, replicas):
    """batch_size is per replica; accumulation is derived, never a CLI input."""
    if min(effective_batch, batch_size, replicas) < 1:
        raise ValueError("effective batch, batch size and replica count must be positive")
    global_microbatch = batch_size * replicas
    if effective_batch % global_microbatch:
        raise ValueError("effective batch must be divisible by batch-size * replicas")
    return {"effective_batch_size": int(effective_batch),
            "microbatch_size": int(batch_size), "replicas": int(replicas),
            "global_microbatch_size": int(global_microbatch),
            "gradient_accumulation_steps": int(effective_batch // global_microbatch)}


def prefetch_batches(iterable, depth=2):
    """One producer owns the iterator; exceptions propagate and early exit joins it.

    Futures hold at most `depth` pending batches. No producer thread can block on
    a full queue when training stops early. Keep depth=0 for SQLite connections
    created on the main thread.
    """
    if depth < 0:
        raise ValueError("prefetch depth cannot be negative")
    if depth == 0:
        yield from iterable
        return
    iterator = iter(iterable)
    sentinel = object()
    def fetch():
        return next(iterator, sentinel)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mind-prefetch") as pool:
        pending = deque(pool.submit(fetch) for _ in range(depth))
        try:
            while pending:
                result = pending.popleft().result()
                if result is sentinel:
                    return
                pending.append(pool.submit(fetch))
                yield result
        finally:
            for future in pending:
                future.cancel()


def effective_batches(groups, size):
    """Join impression-group tails without changing order or losing candidates."""
    import numpy as np
    parts, labels, metadata = [], [], []
    count = 0
    for features, y, rows in groups:
        if not bool(y["labeled"].all()):
            raise ValueError("train iterator returned unlabeled rows")
        start = 0
        while start < len(rows):
            stop = min(len(rows), start + size - count)
            parts.append({key: value[start:stop] for key, value in features.items()})
            labels.append(y["click"][start:stop])
            metadata.extend(rows[start:stop])
            count += stop - start
            start = stop
            if count == size:
                yield ({key: np.concatenate([p[key] for p in parts]) for key in parts[0]},
                       np.concatenate(labels).astype(np.float32), metadata)
                parts, labels, metadata, count = [], [], [], 0
    if count:
        yield ({key: np.concatenate([p[key] for p in parts]) for key in parts[0]},
               np.concatenate(labels).astype(np.float32), metadata)
