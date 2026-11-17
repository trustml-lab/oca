from datasets import load_from_disk

score = load_from_disk("/home/jyj1025/osg-advbandit/data/nq/qwen3-8B/score/final")
label = load_from_disk("/home/jyj1025/osg-advbandit/data/nq/qwen3-8B/labeled")

score = score.remove_columns("label")
merged = score.add_column("label", label["label"])

merged.save_to_disk("/home/jyj1025/osg-advbandit/data/nq/qwen3-8B", max_shard_size="32MB")