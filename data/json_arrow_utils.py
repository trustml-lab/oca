import glob
import json
import os
import re

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset


def natural_sort_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def sorted_json_files(input_dir):
    if os.path.isfile(input_dir):
        if not input_dir.endswith(".json"):
            raise ValueError(f"Expected a .json file: {input_dir}")
        return [input_dir]
    return sorted(glob.glob(os.path.join(input_dir, "*.json")), key=natural_sort_key)


def load_json_records(input_dir):
    records = []
    for path in sorted_json_files(input_dir):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        records.extend(data)
    return records


def add_row_ids(records, start=0):
    for row_id, record in enumerate(records, start=start):
        record["row_id"] = row_id
    return records


def load_arrow_dataset(input_dir):
    arrow_files = sorted(glob.glob(os.path.join(input_dir, "data-*.arrow")), key=natural_sort_key)
    if not arrow_files:
        raise FileNotFoundError(f"No data-*.arrow files found in {input_dir}")

    tables = []
    for path in arrow_files:
        with open(path, "rb") as f:
            tables.append(ipc.open_stream(f).read_all())

    table = pa.concat_tables(tables)
    table = table.replace_schema_metadata({})
    return Dataset(table)


def arrow_chunk_path(output_dir, chunk_idx):
    return os.path.join(output_dir, f"data-{chunk_idx:05d}.arrow")


def arrow_lock_path(output_dir, chunk_idx):
    return f"{arrow_chunk_path(output_dir, chunk_idx)}.lock"


def claim_arrow_chunk(output_dir, chunk_idx):
    os.makedirs(output_dir, exist_ok=True)
    lock_path = arrow_lock_path(output_dir, chunk_idx)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    return lock_path


def release_arrow_chunk(lock_path):
    if not lock_path:
        return
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def write_arrow_chunk(records, output_dir, chunk_idx):
    if not records:
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = arrow_chunk_path(output_dir, chunk_idx)
    table = pa.Table.from_pylist(records)
    with open(path, "wb") as f:
        with ipc.new_stream(f, table.schema) as writer:
            writer.write_table(table)
    return path


def iter_chunks(records, chunk_size):
    if chunk_size <= 0:
        chunk_size = max(1, len(records))
    for chunk_idx, start in enumerate(range(0, len(records), chunk_size)):
        yield chunk_idx, start, records[start : start + chunk_size]
