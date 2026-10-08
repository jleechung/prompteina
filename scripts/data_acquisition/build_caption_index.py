"""Build the memory-mapped caption index used by binder training.

Reads the combined TedA+TedB caption CSV and writes:
    <csv_path>.blob         all description/alignment/facts strings, concatenated as UTF-8
    <csv_path>.offsets.pkl  per-key (offset, length) tables into the blob

The captions carry bidirectional interface H-bonds, hydrophobic contacts and
their breakdowns, plus the alignments the alignment stage reads
(load_alignments=true).

Usage:
    DATA_PATH=/path/to/prompteina-data \
        python scripts/data_acquisition/build_caption_index.py

CSV_PATH may also be set directly to point at a caption CSV elsewhere.
"""

import functools
import os
import pickle
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

print = functools.partial(print, flush=True)

CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(
        os.environ.get("DATA_PATH", "./data"), "captions", "dimers.csv"
    ),
)
BLOB_PATH = CSV_PATH + ".blob"
OFFSETS_PATH = CSV_PATH + ".offsets.pkl"
NODE_RAM_GB = 1600


def main():
    if os.path.exists(OFFSETS_PATH) and os.path.exists(BLOB_PATH):
        print(f"Index already exists: {OFFSETS_PATH} + {BLOB_PATH}")
        print("Delete manually to regenerate.")
        sys.exit(0)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found: {CSV_PATH}", file=sys.stderr)
        print("Download the caption CSVs into $DATA_PATH/captions/ first.",
              file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    desc_acc = {}
    align_acc = {}
    facts_acc = {}
    pos = 0

    os.makedirs(os.path.dirname(BLOB_PATH), exist_ok=True)
    blob = open(BLOB_PATH, "wb", buffering=1024 * 1024)

    def write(s):
        nonlocal pos
        b = (s or "").encode("utf-8")
        off = pos
        blob.write(b)
        pos += len(b)
        return off, len(b)

    csv_size = os.path.getsize(CSV_PATH) / 1e9
    print(f"Reading {CSV_PATH} ({csv_size:.1f} GB)")
    ro = pa_csv.ReadOptions(block_size=64 * 1024 * 1024)
    po = pa_csv.ParseOptions(newlines_in_values=True)

    # Probe header for available columns
    with open(CSV_PATH) as f:
        header = f.readline().strip().split(",")
    has_facts = "facts_json" in header
    has_align = "alignments_json" in header

    include = ["accession", "description"]
    if has_align:
        include.append("alignments_json")
    if has_facts:
        include.append("facts_json")

    co = pa_csv.ConvertOptions(
        include_columns=include,
        column_types={col: pa.string() for col in include},
    )
    reader = pa_csv.open_csv(CSV_PATH, read_options=ro, parse_options=po, convert_options=co)

    total = 0
    for batch in reader:
        accs = batch.column("accession").to_pylist()
        descs = batch.column("description").to_pylist()
        aligns = batch.column("alignments_json").to_pylist() if has_align else [None] * len(accs)
        facts_list = batch.column("facts_json").to_pylist() if has_facts else [None] * len(accs)
        for acc, desc, align, facts in zip(accs, descs, aligns, facts_list):
            total += 1
            if not acc:
                continue
            key = acc.strip().lower()
            desc_acc.setdefault(key, []).append(write(desc))
            if has_align and align:
                align_acc.setdefault(key, []).append(write(align))
            if has_facts and facts:
                facts_acc.setdefault(key, []).append(write(facts))
        elapsed = time.time() - t0
        if total % 5_000_000 == 0:
            print(
                f"  {total:>12,} rows, {len(desc_acc):,} accessions, "
                f"blob {pos / 1e9:.1f} GB ({total / elapsed:,.0f} rows/s)"
            )

    blob.close()
    print(f"\nBlob written: {pos / 1e9:.1f} GB, {len(desc_acc):,} accessions, {total:,} rows")

    # Pack offset tables
    print("Packing offset tables to numpy...")
    t2 = time.time()
    desc_tbl = {k: np.asarray(v, dtype=np.int64) for k, v in desc_acc.items()}
    tables = {"desc": desc_tbl}
    if has_align:
        align_tbl = {k: np.asarray(v, dtype=np.int64) for k, v in align_acc.items()}
        tables["align"] = align_tbl
    if has_facts:
        facts_tbl = {k: np.asarray(v, dtype=np.int64) for k, v in facts_acc.items()}
        tables["facts"] = facts_tbl
    del desc_acc, align_acc, facts_acc
    print(f"  packed in {time.time() - t2:.0f}s")

    print(f"Writing offsets: {OFFSETS_PATH}")
    with open(OFFSETS_PATH, "wb") as f:
        pickle.dump(tables, f, protocol=pickle.HIGHEST_PROTOCOL)

    off_gb = os.path.getsize(OFFSETS_PATH) / 1e9
    blob_gb = os.path.getsize(BLOB_PATH) / 1e9
    print(f"  offsets {off_gb:.2f} GB, blob {blob_gb:.1f} GB")
    print(f"\nMemory footprint (8-GPU DDP / 1 node):")
    print(f"  per-rank offset table ~= {off_gb:.2f} GB on-disk (in-RAM ~2-3x)")
    print(f"  blob {blob_gb:.0f} GB mmap'd = ONE shared page-cache copy per node")
    print(f"  8 ranks ~= {8 * off_gb * 3:.0f} GB offsets + {blob_gb:.0f} GB blob "
          f"vs {NODE_RAM_GB} GB ceiling")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
