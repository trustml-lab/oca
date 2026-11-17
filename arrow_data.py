import os
import glob
import json
from datasets import load_from_disk, Value, concatenate_datasets
import openai
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.ipc as ipc

root = "/home/jyj1025/osg-advbandit/data_scratch/triviaqa"
gpt_dir = os.path.join(root, "gpt", "generations", "samples")
llama_dir = os.path.join(root, "llama3.1-8B", "llama3.1-8B-sampled")

final_dir = os.path.join(root, "triviaqa_data")
gpt_output_dir = os.path.join(final_dir, "gpt")
llama_output_dir = os.path.join(final_dir, "llama")
os.makedirs(gpt_output_dir, exist_ok=True)
os.makedirs(llama_output_dir, exist_ok=True)

jsonl_path = "/home/jyj1025/osg-advbandit/data/triviaqa/qna.jsonl"
try:
    with open(jsonl_path, "r") as f:
        line_count = sum(1 for _ in f)
    print(f"[JSONL] {os.path.basename(jsonl_path)} length: {line_count}")
except FileNotFoundError:
    print(f"[JSONL] {jsonl_path} not found")
    
gpt_ds_list = []
for i in range(1,11):
    chunk_dir = os.path.join(gpt_dir, f"sample_{i}")
    if not os.path.isdir(chunk_dir):
        print(f"{chunk_dir} doesn;t exist")
        continue 
    gpt_ds = load_from_disk(chunk_dir)
    print(f"[GPT] chunk_{i} dataset length: {len(gpt_ds)}")
    gpt_ds_list.append(gpt_ds)
    
gpt_sampled_ds = concatenate_datasets(gpt_ds_list)
print(f"[GPT] Total concatenated dataset length: {len(gpt_sampled_ds)}")

gpt_label_dir = "/home/jyj1025/osg-advbandit/data_scratch/triviaqa/gpt/generations/labeled/labeled_final"
gpt_label_ds = load_from_disk(gpt_label_dir)
print(f"[GPT] Total labeled dataset length: {len(gpt_label_ds)}")


llama_gen_dir = "/home/jyj1025/osg-advbandit/data_scratch/triviaqa/llama3.1-8B/generations"
llama_gen_ds = load_from_disk(llama_gen_dir)
print(f"[llama] Total dataset length: {len(llama_gen_ds)}")

llama_ds_list = []
for chunk_name in os.listdir(llama_dir):
    chunk_dir = os.path.join(llama_dir, chunk_name)
    if not os.path.isdir(chunk_dir) or not chunk_name.startswith("confidence_chunk"):
        continue
    llama_ds = load_from_disk(chunk_dir)
    print(f"[llama] {chunk_name} dataset length: {len(llama_ds)}")
    llama_ds_list.append(llama_ds)

llama_sampled_ds = concatenate_datasets(llama_ds_list)
print(f"[LLaMA] Total concatenated dataset length: {len(llama_sampled_ds)}")

llama_label_dir = "/home/jyj1025/osg-advbandit/data_scratch/triviaqa/llama3.1-8B/llama3.1-8B-labeled"
llama_label_ds = load_from_disk(llama_label_dir)
print(f"[llama] Total labeled dataset length: {len(llama_label_ds)}")

