import os, sys
import json
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    AdamW,
    get_scheduler,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    GPT2LMHeadModel,
    AutoModelForSequenceClassification
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


import torch.nn as nn
import torch as tc
from tqdm import tqdm
from torch.nn.functional import cosine_similarity, normalize
from datasets import load_metric, Dataset
import numpy as np


import pickle
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path

# from models import ConformalLG, QnALLMWrapper
import models
from torch.utils.data import DataLoader
# Check if the minimal version of Transformers is not installed.
check_min_version("4.21.0.dev0")


class SamplingLG(models.ConformalLG):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def generate_onestep(self, kwargs):
        kwargs_processed = self.base_model.model.prepare_inputs_for_generation(**kwargs)
        for k, v in kwargs_processed.items():
            kwargs[k] = v
        output = self.forward(kwargs)

        
        # decison
        probs = output['logprobs'].exp()
        if kwargs['generation_type'] == 'greedy':
            # greedy decoding
            gen_id = tc.argmax(probs, dim=-1, keepdim=True)
        elif kwargs['generation_type'] == 'sample':
            probs = nn.functional.softmax(probs, dim=-1)
            gen_id = torch.multinomial(probs, num_samples=1)#.squeeze(1)

            
        elif kwargs['generation_type'] == 'labeled':
            gen_id = kwargs['current_answer_ids'].unsqueeze(1)
        else:
            raise NotImplementedError
            

        return {'gen_id': gen_id, **output}
    

def main():
    torch.cuda.empty_cache()
    root_dir = os.path.join(Path.home(), 'sg-llm/data/nli/qa2d_gpt3.5/sampling')
    split_file_list = ['Z_U.json', 'Z_E.json']#['nq_Z_u.json', 'Z_E_updated.json']
    model_name_or_path = 'microsoft/deberta-v2-xxlarge-mnli'
    
    # Set seed before initializing model.
    set_seed(42)
    sample_size = 10
    batch_size = 64

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16, # For speed
        #device_map='auto',
        #trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)#, use_fast=False)
    if model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(config, 'max_position_embeddings'):
        tokenizer.max_length = config.max_position_embeddings
    else:
        tokenizer.max_length = tokenizer.model_max_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = SamplingLG(base_model=models.QnALLMWrapper(base_model, tokenizer), generation_type='sample', gen_len=50)
    model = models.NliLLMWrapper(base_model, tokenizer)
    model.to(device)
    model.eval()


    def tokenize_function(examples):
        tokenized_greedy = tokenizer(examples['greedy'], add_special_tokens=False) # greedy
        tokenized_sample = tokenizer(examples['sample'], add_special_tokens=False)
        # labels = examples['labels']
        if model_name_or_path.startswith('potsawee'):
            tokenized_greedy, tokenized_sample = tokenized_sample, tokenized_greedy

        # TODO MJ: added temporary value -1 to labels. Should be modified if the model will be trained.
        result = {
            'input_ids': [[tokenizer.bos_token_id] + s1 + [tokenizer.sep_token_id] + s2 + [tokenizer.eos_token_id] for s1, s2 in zip(tokenized_greedy['input_ids'], tokenized_sample['input_ids'])],
            'attention_mask': [[1] + s1 +[1] + s2 + [1] for s1, s2 in zip(tokenized_greedy['attention_mask'], tokenized_sample['attention_mask'])],
            'token_type_ids': [[0]*(1 + len(s1) +1) + [1]*(len(s2) + 1) for s1, s2 in zip(tokenized_greedy['input_ids'], tokenized_sample['input_ids'])],
        }

        max_length = tokenizer.max_length
        input_ids_list = result['input_ids']
        attention_mask_list = result['attention_mask']
        token_type_ids_list = result['token_type_ids']

        # truncate
        input_ids_list = [i[-max_length:] for i in input_ids_list]
        attention_mask_list = [i[-max_length:] for i in attention_mask_list]
        token_type_ids_list = [i[-max_length:] for i in token_type_ids_list]

        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'token_type_ids': token_type_ids_list,
        }
    
    def collate_fn(batch):
        max_length_batch = max(len(b['input_ids']) for b in batch)
        max_length = tokenizer.max_length
        pad_token_id = tokenizer.pad_token_id

        for i in range(len(batch)):
            assert len(batch[i]['input_ids']) == len(batch[i]['attention_mask']), \
                f"{len(batch[i]['input_ids'])} == {len(batch[i]['attention_mask'])}"
            # right padding
            n_pad = max_length_batch - len(batch[i]['input_ids'])

            batch[i]['input_ids'] = tc.tensor(batch[i]['input_ids'] + [pad_token_id]*n_pad)
            batch[i]['attention_mask'] = tc.tensor(batch[i]['attention_mask'] + [0]*n_pad)
            batch[i]['token_type_ids'] = tc.tensor(batch[i]['token_type_ids'] + [pad_token_id]*n_pad)
            # batch[i]['labels'] = tc.Tensor([batch[i]['labels']])

        # labels = tc.tensor([b['labels'] for b in batch])
        batch = {
            'input_ids': tc.vstack([b['input_ids'] for b in batch]),
            'attention_mask': tc.vstack([b['attention_mask'] for b in batch]),
            'token_type_ids': tc.vstack([b['token_type_ids'] for b in batch]),
        }

        return batch#, labels#, answer_mask, context_mask
    

    def expand_samples(example):
        gen_seq = [example['generated_sequence']] * len(example["samples"])
        return {"greedy": gen_seq, "sample": example["samples"]}


    for split_file in split_file_list:
        data_json = json.load(open(os.path.join(root_dir, split_file), 'r'))

        expanded_data_json = []
        for data in data_json:
            q = 'Question:\n' + data['question'] + '\n\nAnswer:\n'
            gen = data['generated_sequence']
            for sample in data['samples']:
                expanded_data_json.append(
                    {
                        'question': q,
                        'greedy': gen,
                        'sample': sample,
                    }
                )

        # data_json = temp
        raw_dataset = Dataset.from_list(expanded_data_json)

        # print(raw_dataset)
        # raw_dataset = raw_dataset.map(
        #     expand_samples,
        #     batched=True,
        #     num_proc=64,
        #     remove_columns=["samples", "generated_sequence"],
        #     desc="expanding samples",
        # )

        # print(raw_dataset)
        # print(raw_dataset['greedy'][0])
        # print(raw_dataset['sample'][0])
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=64,
            remove_columns=raw_dataset.column_names,
            desc="running tokenization"
        )

        dataloader = DataLoader(
            tokenized_dataset,
            collate_fn=collate_fn,
            batch_size=batch_size*sample_size,
            shuffle=False,
            num_workers=64
        )
        for idx, batch in enumerate(tqdm(dataloader)):
            with tc.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}

                output = model(batch)
                probs = tc.softmax(output['logits'], dim=-1)

                # print(probs.shape)
                
                for jdx in range(probs.shape[0]//sample_size):
                    data_json[idx*batch_size + jdx]['samples_scores'] = probs[jdx*sample_size:(jdx+1)*sample_size, :].tolist()

            # break

        json.dump(data_json, open(os.path.join(root_dir, 'sampling_with_scores', split_file), 'w'), indent=4)
            # sys.exit()
        # json.dump(nq_dataset, open(sys.argv[1], 'w'), indent=4)
        # write
        #data_jsonl = '\n'.join([json.dumps(d) for d in data])
        #open(os.path.join(root_dir, 'nq_udep.jsonl'), 'w').write(data_jsonl)

if __name__== '__main__':
    main()
