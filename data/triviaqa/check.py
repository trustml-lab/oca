import json
from datasets import load_from_disk, DatasetDict

qna_path = '/home/jyj1025/osg-advbandit/data/triviaqa/qna.jsonl'
remove_set = set()
with open(qna_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if item.get('whether_transformed') is False:
            remove_set.add(item['q'])
print(f'{len(remove_set)} will be removed')

arrow_dir = '/home/jyj1025/osg-advbandit/data/triviaqa/gpt-3.5-revised'
ds = load_from_disk(arrow_dir)
print(f'previous {len(ds)} question records loaded')

ds_filtered = ds.filter(lambda ex: ex['question'] not in remove_set)
print(f'after update {len(ds_filtered)} records remain')

output_dir = '/home/jyj1025/osg-advbandit/data/triviaqa/gpt-3.5'
ds_filtered.save_to_disk(output_dir, max_shard_size="32MB")
print(f'data updated at {output_dir}')
