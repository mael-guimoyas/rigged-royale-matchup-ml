"""Split each prepared split's single data.parquet into N parquet files.

pyarrow.dataset exposes one fragment per file, and the training dataloader
shards fragments across worker processes. A single file means a single fragment,
so num_workers>0 cannot parallelise the (CPU-bound) per-row encoding. Splitting
into N files lets N workers encode concurrently with no change to the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PREP = Path("data/prepared")
SPLITS = ["train", "validation", "test"]
N = 12
SCAN = 65_536


def repartition(split: str) -> None:
    d = PREP / split
    src = d / "data.parquet"
    if not src.exists():
        print(f"{split}: no data.parquet, skip")
        return
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    total_src = pf.metadata.num_rows
    parts = [d / f"part-{i:03d}.parquet" for i in range(N)]
    writers = [pq.ParquetWriter(p, schema, compression="zstd") for p in parts]
    rows = 0
    try:
        for i, rb in enumerate(pf.iter_batches(batch_size=SCAN)):
            writers[i % N].write_table(pa.Table.from_batches([rb]))
            rows += rb.num_rows
    finally:
        for w in writers:
            w.close()
        pf.close()
    written = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    if written != total_src:
        raise SystemExit(f"{split}: row mismatch src={total_src} written={written}; kept data.parquet")
    src.unlink()
    print(f"{split}: {total_src} rows -> {N} files, data.parquet removed")


def main() -> None:
    for split in SPLITS:
        repartition(split)


if __name__ == "__main__":
    sys.exit(main())
