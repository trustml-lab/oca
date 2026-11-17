import os
import json
from datasets import Dataset

score_dir = "/home/jyj1025/osg-advbandit/data/nq/qwen3-8B/score/scratch2"
json_files = [
    os.path.join(score_dir, fn)
    for fn in os.listdir(score_dir)
    if fn.endswith("_per_question.json")
]

merged_list = []
for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        merged_list.extend(json.load(f))

merged_json_path = os.path.join(score_dir, "score.json")
with open(merged_json_path, "w", encoding="utf-8") as f:
    json.dump(merged_list, f, ensure_ascii=False, indent=2)
print("score.json saved")

ds = Dataset.from_list(merged_list)
output_dir = os.path.join(score_dir, "final")
ds.save_to_disk(output_dir, max_shard_size="32MB")
