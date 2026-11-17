# entailment oracle을 사용하는 부분 분리 필요 

import json
import math
import os
import csv
import openai
import matplotlib.pyplot as plt

client = openai.OpenAI(api_key = '')

def language_generator(
    messages, 
    model="gpt-3.5-turbo",   
    max_tokens=500,               
    temperature=0,           
    stop=None,
    seed=123,      
    tools=None,
    logprobs=True,  
):
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "logprobs": logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion

def entailment_oracle(seq1, seq2):
    checking_prompt = f"""
        You are an expert in Natural Language Inference. Your task is to determine whether the first sentence contradicts the second sentence.

        Please provide an answer with a score of 0 or 1, where:
        1 = contradiction
        0 = no contradiction

        The first sentence: {seq1}
        The second sentence: {seq2}

        Provide your score only with number.
        Score:
    """
    result = language_generator(
        [{"role": "user", "content": checking_prompt}],
        model="gpt-3.5-turbo", # other model  (4o, 4o-mini)
        max_tokens=10,
        temperature=0
    )
    score = result.choices[0].message.content.strip()
    return int(score) if score.isdigit() else 0

def main():
    with open("test_data/test_sample.json", "r") as f:
        data = json.load(f)
    
    results = []
    
    for qa_data in data:        
        question = qa_data["question"]
        answer = qa_data.get("transformed")
        whether_transformed = qa_data.get("whether_transformed")
        
        # Step 1: generate response
        API_RESPONSE = language_generator(
            [{"role": "user", "content": question}],
            model="gpt-3.5-turbo", 
            logprobs=True,
        )
        response_text = API_RESPONSE.choices[0].message.content.strip()
    
        # Step 2: get confidence score
        logprobs = [token.logprob for token in API_RESPONSE.choices[0].logprobs.content]
        sum_of_logprobs = sum(logprobs)
        average_logprob = sum_of_logprobs / len(logprobs)
        confidence_score = math.exp(average_logprob)
    
        # Step 3: access entailment oracle
        contradition_flag = entailment_oracle(response_text, answer)
        
        IDK_flag = None
        
        result = {
            "question": question,
            "response_text": response_text,
            "answer": answer,
            "current_score": confidence_score,
            "contradition_flag": contradition_flag,
            "IDK_flag": IDK_flag,
            "whether_transformed": whether_transformed
        }
        
        results.append(result)
        
    output_path = "test_data/test_sample_dict.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
