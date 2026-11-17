###############################################################################
# Not the complete code, but refer to the prompt we used to generate samples. #
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
    temperature=1,
    stop=None,
    seed=123,
    tools=None,
    logprobs=True,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=None,
    n=10,
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
        "n": n,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion

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
                model="gpt-3.5-turbo",
                logprobs=False,
                n=10,
                # top_logprobs=5,
            )
            break
        except Exception as e:
            print("retrying..")
            if attempt == 4:
                raise e
            time.sleep(30)
            continue
    # logprobs = [token.logprob for token in API_RESPONSE.choices[0].logprobs.content]
    response_text = API_RESPONSE.choices[0].message.content
    # print(response_text)
    data['samples'] = [choice.message.content for choice in API_RESPONSE.choices]
    print(data['samples'])
    # output.append(
    #     {
    #         "question":data['question'],
    #         "answer":data['answer'],
    #         "generated_sequence":response_text,
    #         "transformed_sequence":data['transformed'],
    #         # "logprobs": logprobs,
    #     }
    # )
    
json.dump(data_json, open(f'data/nq/gpt-3.5-turbo/samples/nq_sam{start//20000}.json', 'w'), indent=4)



