import os
import argparse
import re
import time
import openai
from tqdm.auto import tqdm
try:
    from json_arrow_utils import (
        add_row_ids,
        arrow_chunk_path,
        claim_arrow_chunk,
        iter_chunks,
        load_arrow_dataset,
        load_json_records,
        release_arrow_chunk,
        sorted_json_files,
        write_arrow_chunk,
    )
except ImportError:
    from data.json_arrow_utils import (
        add_row_ids,
        arrow_chunk_path,
        claim_arrow_chunk,
        iter_chunks,
        load_arrow_dataset,
        load_json_records,
        release_arrow_chunk,
        sorted_json_files,
        write_arrow_chunk,
    )

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
DEFAULT_DATA_DIR = "/home/jyj1025/osg-advbandit/data/nq/gpt-5.4/generations"
DEFAULT_OUTPUT_DIR = "/home/jyj1025/osg-advbandit/data/nq/gpt-5.4/labels-gpt-5.4-nano"
SYSTEM_INSTRUCTIONS = (
    "You are an expert in Natural Language Inference. "
    # "Your task is to determine whether the answer_sequence entails or contradicts the generated_sequence."
    "Determine whether the answer_sequence entails or contradicts the generated_sequence. "
    "Return exactly one integer and nothing else: 1 for entailment, 0 for contradiction."
)


def load_dataset_from_dir(generations_dir):
    return load_arrow_dataset(generations_dir)


def require_columns(ds, columns):
    missing = [column for column in columns if column not in ds.column_names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_openai_client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required.")
    return openai.OpenAI()


def extract_response_text(resp):
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    parts = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if isinstance(content, dict):
                content_text = content.get("text")
            else:
                content_text = getattr(content, "text", None)
            if content_text:
                parts.append(content_text)
    return "\n".join(parts).strip()


def parse_binary_label(raw):
    match = re.search(r"\b[01]\b", raw.strip())
    return int(match.group(0)) if match else -1


def detect_input_format(input_dir):
    if sorted_json_files(input_dir):
        return "json"
    try:
        load_arrow_dataset(input_dir)
        return "arrow"
    except FileNotFoundError:
        raise FileNotFoundError(f"No JSON or data-*.arrow files found in {input_dir}")


def require_json_fields(records, columns):
    for idx, record in enumerate(records):
        missing = [column for column in columns if column not in record]
        if missing:
            raise ValueError(f"Record {idx} is missing required fields: {missing}")


def load_label_json_records(input_dir, row_id_offset=0):
    records = add_row_ids(load_json_records(input_dir), start=row_id_offset)
    for record in records:
        if "transformed_sequence" not in record and "transformed" in record:
            record["transformed_sequence"] = record["transformed"]
    require_json_fields(records, ["generated_sequence", "transformed_sequence"])
    return records


def load_label_arrow_records(input_dir, row_id_offset=0):
    ds = load_dataset_from_dir(input_dir)
    require_columns(ds, ["generated_sequence", "transformed_sequence"])

    records = []
    for idx, record in enumerate(ds):
        record = dict(record)
        if "row_id" not in record or record["row_id"] is None:
            record["row_id"] = row_id_offset + idx
        records.append(record)
    return records


def is_contradiction(
    client,
    generated_sequence,
    transformed_sequence,
    model=DEFAULT_MODEL,
    max_output_tokens=32,
    reasoning_effort="none",
    max_retries=5,
    fail_on_error=False,
):
    prompt = f"""generated_sequence:
{generated_sequence}

answer_sequence:
{transformed_sequence}

Score:
"""

    max_retries = max(1, max_retries)
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=prompt,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_output_tokens,
                store=False,
            )
            label = parse_binary_label(extract_response_text(resp))
            if label in (0, 1):
                return label
            raise ValueError("OpenAI response did not contain a binary label.")
        except Exception as exc:
            if attempt == max_retries - 1:
                if fail_on_error:
                    raise
                print(f"Labeling failed after {max_retries} attempt(s): {exc}")
                return -1
            time.sleep(2 ** attempt)


def label_records(client, records, args, source_name="records"):
    os.makedirs(args.output_dir, exist_ok=True)
    chunks_written = 0
    chunks_seen = 0

    for local_chunk_idx, _, chunk in iter_chunks(records, args.chunk_size):
        chunk_idx = local_chunk_idx + args.output_chunk_offset
        if local_chunk_idx < args.start_chunk:
            continue
        if args.max_chunks is not None and chunks_seen >= args.max_chunks:
            break

        chunks_seen += 1
        output_path = arrow_chunk_path(args.output_dir, chunk_idx)
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping existing chunk {chunk_idx}: {output_path}")
            continue

        lock_path = claim_arrow_chunk(args.output_dir, chunk_idx)
        if lock_path is None:
            print(f"Skipping locked chunk {chunk_idx}: {output_path}")
            continue

        try:
            if os.path.exists(output_path) and not args.overwrite:
                print(f"Skipping existing chunk {chunk_idx}: {output_path}")
                continue

            labeled_chunk = []
            for record in tqdm(chunk, total=len(chunk), desc=f"label chunk {chunk_idx}"):
                labeled_record = dict(record)
                labeled_record["label"] = is_contradiction(
                    client,
                    labeled_record["generated_sequence"],
                    labeled_record["transformed_sequence"],
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                    max_retries=args.max_retries,
                    fail_on_error=args.fail_on_error,
                )
                labeled_chunk.append(labeled_record)

            saved_path = write_arrow_chunk(labeled_chunk, args.output_dir, chunk_idx)
            chunks_written += 1
            print(f"Saved labeled chunk {chunk_idx} ({len(labeled_chunk)} rows) to: {saved_path}")
        finally:
            release_arrow_chunk(lock_path)

    print(f"Finished {source_name} labeling. Wrote {chunks_written} new chunk(s) to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input_format", type=str, default="json", choices=["auto", "arrow", "json"])
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="none",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--max_output_tokens", type=int, default=32)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--start_chunk", type=int, default=0)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--row_id_offset", type=int, default=0)
    parser.add_argument("--output_chunk_offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    client = get_openai_client()
    print(f"Using OpenAI model: {args.model}")

    input_format = args.input_format
    if input_format == "auto":
        input_format = detect_input_format(input_dir)

    if input_format == "json":
        print("////////// JSON Data Loading //////////")
        records = load_label_json_records(input_dir, row_id_offset=args.row_id_offset)
        print(f"total samples count: {len(records)}")
        print(f"Saving labeled Arrow chunks to: {output_dir}")
        label_records(client, records, args, source_name="JSON")
        return

    print("////////// Data Loading //////////")
    records = load_label_arrow_records(input_dir, row_id_offset=args.row_id_offset)
    print(f"total samples count: {len(records)}")
    print(f"Saving labeled Arrow chunks to: {output_dir}")
    label_records(client, records, args, source_name="Arrow")

if __name__ == "__main__":
    main()
