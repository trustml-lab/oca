import json

with open("data/nq/qwen3-8B/score/score_9.json", encoding="utf-8") as f:
    data = json.load(f)
    
keys = list(data.keys())
n = len(data[keys[0]])
records = []
for i in range(n):
    rec = {k: data[k][i] for k in keys}
    records.append(rec)


with open("data/nq/qwen3-8B/score/score_9_per_question.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved {len(records)} records to score_9_per_question.json")
