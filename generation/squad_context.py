from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets import Dataset
import seaborn as sns
import numpy as np
from tqdm import tqdm

# dataset = load_dataset("rajpurkar/squad")
dataset = load_from_disk("data/dialog/squad")


dataset = [{'id': None, 'title': data['title'], 'context': data['context']} for data in dataset]

output = []
init_data = dataset[0]
init_data['id'] = 0
output.append(dataset[0])
last_context = dataset[0]['context']
last_title = dataset[0]['title']
current_index = 0
for data in tqdm(dataset[1:]):
    # print(data)
    if data['title'] != last_title:
        current_index += 1
        last_title = data['title']
    if data['context'] != last_context:
        data['id'] = current_index
        output.append(data)
        last_context = data['context']

output = Dataset.from_list(output)
output.save_to_disk("data/dialog/squad/contexts", max_shard_size="32MB")


