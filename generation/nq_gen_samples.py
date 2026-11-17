###############################################################################
# Not the complete code, but refer to the prompt we used to generate answers. #
###############################################################################


import os, sys
import json
import logging
import math
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
    # HfArgumentParser,
    # Trainer,
    # TrainingArguments,
    # default_data_collator,
    # is_torch_tpu_available, # deprecated in the newer version
    set_seed,
    # AdamW, # deprecated in the newer version
    # get_scheduler,
    # DataCollatorWithPadding,
    # DataCollatorForLanguageModeling,
    # DataCollatorForSeq2Seq,
    # TrainerCallback,
    # GPT2LMHeadModel,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


import torch.nn as nn
import torch as tc
from tqdm import tqdm
from torch.nn.functional import cosine_similarity, normalize
from datasets import Dataset
import numpy as np


import pickle
from typing import Callable, Dict, List, NewType, Optional, Tuple, Union #, Any
from tqdm import tqdm
from pathlib import Path

from models import ConformalLG, QnALLMWrapper, ConformalLG
from torch.utils.data import DataLoader
from models.util import pad_left


# class SamplingLG(ConformalLG):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#     def generate_onestep(self, kwargs):
#         kwargs_processed = self.base_model.model.prepare_inputs_for_generation(**kwargs)
#         for k, v in kwargs_processed.items():
#             kwargs[k] = v
#         output = self.forward(kwargs)

        
#         # decison
#         probs = output['logprobs'].exp()
#         if kwargs['generation_type'] == 'greedy':
#             # greedy decoding
#             gen_id = tc.argmax(probs, dim=-1, keepdim=True)
#         elif kwargs['generation_type'] == 'sample':
#             probs = nn.functional.softmax(probs, dim=-1)
#             gen_id = torch.multinomial(probs, num_samples=1)#.squeeze(1)

            
#         elif kwargs['generation_type'] == 'labeled':
#             gen_id = kwargs['current_answer_ids'].unsqueeze(1)
#         else:
#             raise NotImplementedError
            

#         return {'gen_id': gen_id, **output}
    

def main():
    torch.cuda.empty_cache()
    root_dir = os.path.join(Path.home(), 'media1', 'datasets', 'lang', 'nli', 'nq')
    output_path = Path.home() / 'research' / 'osg-advbandit' / 'data' / 'nq_llama3.1-8B' / 'nq_sam.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # split_file_list = ['Z_E_updated.json', 'nq_Z_u.json']
    split_file_list = ['nq_paired_data.json']
    # split_file_list = ['Z_E_updated.json']
    model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Set seed before initializing model.
    set_seed(42)
    sample_size = 10
    batch_size=32

    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id


    # model = SamplingLG(base_model=models.QnALLMWrapper(base_model, tokenizer), generation_type='sample', gen_len=50)
    model = base_model
    model.to(device)
    model.eval()

    def tokenize_function(examples):

        # concat
        question_list = examples['question']
        # MJ: CAUTION!!
        # may not be used except for PEFT
        answer_list = examples['answer']#examples['transformed_answer'] if ('transformed_answer' in examples) and (training_args.prompt_model_path is None) else examples['answer']

        # tokenize
        # tokenized_questions = tokenizer(question_list, add_special_tokens=False)
        tokenized_answers = tokenizer(answer_list, add_special_tokens=False)

        tokenized_prompts = [
            tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': "You are a helpful AI assistant for question answering. Answer the question in a single sentence."},
                    {'role': 'user', 'content': question}
                ],
                add_generation_prompt=True
            )
            for question in question_list
        ]

        output = {
            'input_ids': [q for q, a in zip(tokenized_prompts, tokenized_answers['input_ids'])],
            'attention_mask': [[1]*len(q) for q, a in zip(tokenized_prompts, tokenized_answers['attention_mask'])],
            # 'answer_mask': [[0]*(len(q)) + a + [1] for q, a in zip(tokenized_prompts, tokenized_answers['attention_mask'])],
        }

        return output
    
    def collate_fn(batch):
        max_length_batch = max(len(b['input_ids']) for b in batch)
        #max_length = tokenizer.max_length
        pad_token_id = tokenizer.pad_token_id
        # print(tokenizer.pad_token_id)
        input_ids = []
        attention_mask = []
        answer_mask = []
        for i in range(len(batch)):
            assert len(batch[i]['input_ids']) == len(batch[i]['attention_mask']), \
                f"{len(batch[i]['input_ids'])} == {len(batch[i]['attention_mask'])}"
            
            # INFO: left padding
            n_pad = max_length_batch - len(batch[i]['input_ids'])
            input_ids.append(tc.tensor([pad_token_id]*n_pad + batch[i]['input_ids']))
            attention_mask.append(tc.tensor([0]*n_pad + batch[i]['attention_mask']))
            # answer_mask.append(tc.tensor([0]*n_pad + batch[i]['answer_mask'])) # 0
        input_ids = tc.vstack(input_ids)
        attention_mask = tc.vstack(attention_mask)

        # answer_mask = tc.vstack(answer_mask)

        # label = answer_mask

        # return
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask}#, 'answer_mask': answer_mask}
        # label = answer_mask
        
        return batch#, label  #(x, y) format
    
    for split_file in split_file_list:
        data_json = json.load(open(os.path.join(root_dir, split_file), 'r'))#[:30000]
        temp = []
        for data in data_json:
            # print(data)
            if data.pop('whether_transformed'):
                # data['question'] = data['question']
                # data['answer'] = data['answer']

                data['prompt'] = tokenizer.apply_chat_template(
                        [
                            {'role': 'system', 'content': "You are a helpful AI assistant for question answering. Answer the question in a single sentence."},
                            {'role': 'user', 'content': data['question']}
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                temp.append(data)
        data_json = temp

        raw_dataset = Dataset.from_list(data_json)
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
            batch_size=batch_size,
            shuffle=False,
            num_workers=64
        )
        for idx, batch in enumerate(tqdm(dataloader)):
            with tc.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['inputs_embeds'] = model.model.embed_tokens(batch.pop('input_ids'))

                # samples = []
                outputs = model.generate(
                    **batch,
                    max_new_tokens=50,
                    num_return_sequences=sample_size,
                    # no_repeat_ngram_size=2,
                    do_sample=True,
                    # top_k=50,
                    # top_p=0.95,
                    temperature=1,
                    # early_stopping=True
                )
                # print(outputs.shape)
                generated_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                print(generated_sequences)
                # samples.append(generated_sequences)
                # print(outputs.shape)
                for jdx in range(outputs.shape[0]//sample_size):
                    data_json[idx*batch_size + jdx]['question'] = data_json[idx*batch_size + jdx].pop('question')
                    data_json[idx*batch_size + jdx]['answer'] = data_json[idx*batch_size + jdx].pop('answer')
                    data_json[idx*batch_size + jdx]['prompt'] = data_json[idx*batch_size + jdx].pop('prompt')
                    data_json[idx*batch_size + jdx]['samples'] = generated_sequences[jdx*sample_size:(jdx+1)*sample_size]

            # break

        json.dump(data_json, open(output_path, 'w'), indent=4, ensure_ascii=False)
            # sys.exit()
        # json.dump(nq_dataset, open(sys.argv[1], 'w'), indent=4)
        # write
        #data_jsonl = '\n'.join([json.dumps(d) for d in data])
        #open(os.path.join(root_dir, 'nq_udep.jsonl'), 'w').write(data_jsonl)

if __name__== '__main__':
    main()
