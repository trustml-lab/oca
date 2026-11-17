import json
import os
import openai

openai.api_key = ''

def language_generator(
    messages,
    model="gpt-4o",
    max_tokens=5,
    temperature=0
):
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )


def entailment_oracle(seq1: str, seq2: str) -> int:
    prompt = f"""
        You are an expert in Natural Language Inference. Your task is to determine whether the first sentence contradicts the second sentence.

        Please provide an answer with a score of 0 or 1, where:
        1 = contradiction
        0 = no contradiction

        The first sentence: {seq1}
        The second sentence: {seq2}

        Provide your score only with number.
        Score:
    """
    resp = language_generator(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",
        max_tokens=3,
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    return int(text) if text.isdigit() else 0

def main():
    input_path  = "test_data/test_sample_dict.json"
    output_path = "test_data/test_sample_dict_update.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        response_text = entry.get("response_text", "")
        answer_text   = entry.get("answer", "")
        new_flag = entailment_oracle(response_text, answer_text)

        if "contradition_flag" in entry:
            del entry["contradition_flag"]
        entry["contradiction_flag"] = new_flag

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Updated contradiction flags saved to {output_path}")

if __name__ == "__main__":
    main()
