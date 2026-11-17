import argparse
import os

try:
    from json_arrow_utils import iter_chunks, load_arrow_dataset, write_arrow_chunk
except ImportError:
    from data.json_arrow_utils import iter_chunks, load_arrow_dataset, write_arrow_chunk


DEFAULT_LABELS_DIR = "/home/jyj1025/osg-advbandit/data/nq/gpt-5.4/labels-gpt-5.4-nano"
DEFAULT_SCORES_DIR = "/home/jyj1025/osg-advbandit/data/nq/gpt-5.4/scores-gpt-5.4-nano"
DEFAULT_OUTPUT_DIR = "/home/jyj1025/osg-advbandit/data/nq/gpt-5.4/final-gpt-5.4-nano"


def dataset_records_by_row_id(ds, name):
    if "row_id" not in ds.column_names:
        raise ValueError(f"{name} dataset must contain row_id.")

    records = {}
    for record in ds:
        row_id = record["row_id"]
        if row_id in records:
            raise ValueError(f"Duplicate row_id {row_id} in {name} dataset.")
        records[row_id] = record
    return records


def ensure_output_dir(output_dir):
    existing = []
    if os.path.isdir(output_dir):
        existing = [
            path
            for path in sorted(os.listdir(output_dir))
            if path.endswith(".json") or (path.startswith("data-") and path.endswith(".arrow"))
        ]
    if existing:
        raise FileExistsError(
            f"{output_dir} already contains output-looking files. "
            "Use a fresh output_dir to avoid mixing stale Arrow chunks."
        )
    os.makedirs(output_dir, exist_ok=True)


def merge_records(label_records, score_records):
    missing_scores = sorted(set(label_records) - set(score_records))
    missing_labels = sorted(set(score_records) - set(label_records))
    if missing_scores or missing_labels:
        raise ValueError(
            "Label/score row_id mismatch: "
            f"{len(missing_scores)} label rows without scores, "
            f"{len(missing_labels)} score rows without labels."
        )

    merged = []
    for row_id in sorted(score_records):
        score_record = dict(score_records[row_id])
        label_record = label_records[row_id]

        score_record["label"] = label_record["label"]
        for column in ("generated_sequence", "transformed_sequence", "logprobs"):
            if score_record.get(column) is None and column in label_record:
                score_record[column] = label_record[column]

        merged.append(score_record)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", type=str, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--scores_dir", type=str, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk_size", type=int, default=1000)
    args = parser.parse_args()

    print("////////// Loading labeled/scored Arrow chunks //////////")
    labels_ds = load_arrow_dataset(args.labels_dir)
    scores_ds = load_arrow_dataset(args.scores_dir)
    print(f"label rows: {len(labels_ds)}")
    print(f"score rows: {len(scores_ds)}")

    label_records = dataset_records_by_row_id(labels_ds, "labels")
    score_records = dataset_records_by_row_id(scores_ds, "scores")

    print("////////// Merging //////////")
    merged = merge_records(label_records, score_records)
    ensure_output_dir(args.output_dir)

    for chunk_idx, _, chunk in iter_chunks(merged, args.chunk_size):
        saved_path = write_arrow_chunk(chunk, args.output_dir, chunk_idx)
        print(f"Saved merged chunk {chunk_idx} ({len(chunk)} rows) to: {saved_path}")

    print(f"Saved final merged Arrow dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
