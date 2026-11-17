from datasets import load_dataset, load_from_disk
from datasets import Dataset
import seaborn as sns
import numpy as np


# dataset = load_dataset("squad")

dataset = load_from_disk('data/nq/llama3.1-8B-sampled/confidence_chunk_000')
# print(len(dataset['logprobs']))
# dataset['confidence_score'] = [np.exp(np.array(lp).mean()) for lp in dataset['logprobs']]
# dataset = load_dataset("data/nli", "nq_gpt3.5")


import matplotlib.pyplot as plt
# dataset = dataset['validation']
# confidence_score = ((1-np.array(dataset['samples_scores'])[..., 0]).mean(axis=1)).tolist()
# print(confidence_score)

# dataset = dataset.add_column("confidence_score", confidence_score)
# Convert the dataset to a pandas DataFrame
df = dataset.to_pandas()

# 수평 violin plot: y=label, x=confidence_score
plt.figure(figsize=(10, 4))
sns.violinplot(
    x="confidence_score",
    y="label",
    data=df,
    hue="label",
    orient="h",
    palette={0: "red", 1: "blue"},
    cut=0  # 데이터 바깥의 분포 안 보이게 자름 (선택)
)

plt.title("Confidence Score Distribution by Label")
plt.xlabel("Confidence Score")
plt.ylabel("Label")
plt.yticks([0, 1], ["label 0", "label 1"])
plt.tight_layout()
plt.savefig("violin_horizontal_fm2.png")
plt.show()