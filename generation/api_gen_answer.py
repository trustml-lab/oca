###############################################################################
# Not the complete code, but refer to the prompt we used to generate answers. #
###############################################################################


from openai import OpenAI
import os
import json
from tqdm import tqdm
from pathlib import Path
import time

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=True,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
    top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion


# dataset = 'dataset'
data_json = json.load(open(f'data/nq/nq_paired_data.json', 'r'))
data_json = [d for d in data_json if d['whether_transformed'] == True]
# FIXME Adjust the prompt as needed
QNA_PROMPT = """In a short sentence including the following content, {data}"""


logprobs_data = []
text_data = []
output = []
start = 60000
end = 60000
data_json = data_json[start:]
for idx, data in enumerate(tqdm(data_json)):

    for attempt in range(5):
        try:
            API_RESPONSE = get_completion(
                [{"role": "user", "content": QNA_PROMPT.format(data=data['question'])}],
                model="gpt-3.5-turbo", # FIXME Update the model as needed
                logprobs=True,
                # top_logprobs=5,
            )
            break
        except Exception as e:
            if attempt == 4:
                raise e
            time.sleep(30)
            continue
    logprobs = [token.logprob for token in API_RESPONSE.choices[0].logprobs.content]
    response_text = API_RESPONSE.choices[0].message.content
    print(response_text)

    output.append(
        {
            "question":data['question'],
            "answer":data['answer'],
            "generated_sequence":response_text,
            "transformed_sequence":data['turker_sequence'] if 'turker_sequence' in data else data['transformed'],
            "logprobs": logprobs,
        }
    )
    
json.dump(output, open(f'data/nq/gpt-3.5-turbo/generations/nq_gen{start//30000}.json', 'w'), indent=4)
