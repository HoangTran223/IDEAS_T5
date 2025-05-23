import json
import openai
import random
from openai import OpenAI


API_BASE = "your_api"
API_KEY = "your_api_key"


client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

system_prompt = """Please act as an impartial judge and compare the quality of response A and response B provided by two AI assistants to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. 
Just tell me which response do you think is better:
- If A is significantly better than B, just answer me "A";
- If B is significantly better than A, just answer me "B";
- If A and B have similar quality (both good or both wrong), just answer me "Tied". 

[Question]
{}

[Response A] 
{}

[Response B]
{}
"""

def request_llm_api(content):
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
    )
    return completion


def llm_eval(test_data):
    dskd_win = 0
    vanilla_win = 0
    tied = 0
    data_with_results = []
    for data in test_data:
        rand_var = random.random()
        if rand_var > 0.5:
            ra = data["dskd_answer"]
            rb = data["vanilla_answer"]
        else:
            ra = data["vanilla_answer"]
            rb = data["dskd_answer"]
        
        if data["input"] == "":
            question = data["instruction"]
        else:
            question = data["input"] + "\n" + data["instruction"]

        content = system_prompt.format(question, ra, rb)

        for i in range(5):
            result = request_llm_api(content).choices[0].message.content
            if result not in ["A", "B", "Tied"]:
                continue
            else:
                break
        
        if result not in ["A", "B", "Tied"]: 
            data["compare_result"] = "error"
            continue
        else:
            if result == "Tied":
                tied += 1
                print("tied")
                data["compare_result"] = "tied"
            elif rand_var > 0.5:
                if result == "A":
                    dskd_win += 1
                    data["compare_result"] = "dskd win"
                    print("dskd_win")
                else: 
                    vanilla_win += 1
                    data["compare_result"] = "vanilla win"
                    print("vanilla win")
            elif rand_var <= 0.5:
                if result == "A": 
                    vanilla_win += 1
                    data["compare_result"] = "vanilla win"
                    print("vanilla win")
                else: 
                    dskd_win += 1
                    data["compare_result"] = "dskd win"
                    print("dskd win")
    
        data_with_results.append(data)

    return (dskd_win, vanilla_win, tied), data_with_results

    
if __name__ == "__main__":
    random.seed(42)
    data_num = 10

    dskd_path = "dskd/answers.jsonl"   # path to the responses generated by the DSKD student
    vanilla_path = "vanilla/answers.jsonl"   # path to the responses generated by the vanilla KD student
    ori_data_path = "../../data/dolly/valid.jsonl"
    result_path = "../../llm_judge/kl.json"   # default to forward KL

    with open(ori_data_path) as f:
        data = [json.loads(l) for l in f.readlines()]

    with open(dskd_path) as f:
        dskd_answers = [json.loads(l) for l in f.readlines()]

    with open(vanilla_path) as f:
        vanilla_answers = [json.loads(l) for l in f.readlines()]
    
    test_samples = [
        {
            "instruction": data[i]["instruction"], 
            "input": data[i]["input"],
            "dskd_answer": dskd_answers[i]["text"],
            "vanilla_answer": vanilla_answers[i]["text"]
        } for i in range(len(data))
    ]

    random.shuffle(test_samples)

    eval_results, data_with_results = llm_eval(test_samples[:100])
    print(eval_results)

    json.dump(data_with_results, open(result_path, "w"), indent=2, ensure_ascii=False)


