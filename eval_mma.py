import pandas as pd
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def eval():
    res = []
    for file in eval_folder.rglob("*.json"):
        if file.stem.endswith("_other"):
            continue
        print(file)
        data = json.load(open(file, "r"))
        other = []
        type_count={"Adjective":0, "Attachment Ambiguity":0, "Coordination Ambiguity":0, "Idiom":0, "Noun":0, "Pragmatic Ambiguity":0,"Structural Ambiguity":0,"Verb":0, "overall":0, "other":0}
        total_count={"Adjective":0, "Attachment Ambiguity":0, "Coordination Ambiguity":0, "Idiom":0, "Noun":0, "Pragmatic Ambiguity":0,"Structural Ambiguity":0,"Verb":0, "overall":0, "other":0}
        for i in range(0, len(data),2):
            assert data[i]['class'] == data[i+1]['class'], "Class not equal"
            total_count[data[i]['class']] += 1
            total_count["overall"] += 1
            total_count["other"] += 2

            tar1 = data[i]["target"]
            tar2 = data[i+1]["target"]
            tar1_split = tar1.split('|')[0].strip().upper()
            tar2_split = tar2.split('|')[0].strip().upper()
            ans1 = data[i]["answer"]
            ans2 = data[i+1]["answer"]

            if len(tar1_split) !=1:
                other.append(data[i]) 
                type_count["other"] += 1
            if len(tar2_split) !=1:
                other.append(data[i+1])
                type_count["other"] += 1
            
            if tar1_split == ans1 and tar2_split == ans2:
                type_count["overall"] += 1
                type_count[data[i]['class']] += 1
        json.dump(other, open(file.with_name(file.stem + "_other.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        final = {key: f"{type_count[key]}({total_count[key]})" for key in type_count}
        final["model"] = file.parent.name + file.stem
        res.append(final)
    df = pd.DataFrame(res)
    df.to_csv(eval_folder / "type_count.csv", index=False, encoding='utf-8-sig')

def fix_other():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt_temp = """Extract the selected option (A, B, C, D) from the provided context. Your goal is to identify the chosen option clearly from the context. 
Here are some examples:

Example 1
Context: C: Chris and Pat are playing chess with each other. The image shows a line of boys playing chess, and the names \"Chris\" and \"Pat\" are indicated near each of them. Based on the image, it is reasonable to infer that both Chris and Pat are playing chess with each other, as they are situated next to each other and have the presence of chess pieces on the board in front of them.
Final Answer: C
    
Example 2
Context: Based on the image provided, it appears that Roses and James are having a conversation, but not necessarily an argument. They are standing close to each other, engaged in a discussion. There is no definitive indication in the image that they are actively arguing. The image shows them talking, not fighting, so the most suitable option from the given choices is B: Roses and James are arguing with others respectively.
Final Answer: B

Example 3
Context: Based on the image provided, it is not possible to accurately determine the length requirement of the report and the essay. The text in the image only describes the content of the assignments, not their length. Therefore, I cannot provide an answer to this question as it cannot be inferred from the image alone. 
Final Answer: None

Using the examples above, extract the Final Answer (the selected option) from the provided Context. If no clear choice is possible, return "None."
Context: {mllm_response}
Final Answer: """
    for file in eval_folder.rglob("*_other.json"):
        print(file)
        data = json.load(open(file, "r"))
        for i in tqdm(range(0, len(data))):
            # question = data[i]["source"]
            mllm_response = data[i]["target"]
            # prompt = prompt_temp.format_map({"question": question, "mllm_response": mllm_response})
            prompt = prompt_temp.format(mllm_response=mllm_response)
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            data[i]["target_fix"] = response
            
        json.dump(data, open(file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

def parse_other():
    for file in eval_folder.rglob("*_other.json"): 
        print(file)
        data = json.load(open(file, "r"))
        for item in data:
            target = item["target"]
            if "A:" in target:
                item['target'] = "A | " + target
            elif "B:" in target:
                item['target'] = "B | " + target
            elif "C:" in target:
                item['target'] = "C | " + target
            elif "D:" in target:
                item['target'] = "D | " + target
        json.dump(data, open(file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

def merge_other():

    for file in eval_folder.rglob("*.json"):
        if file.stem.endswith("_other"):
            continue
        print(file)
        data = json.load(open(file, "r"))
        other = json.load(open(file.with_name(file.stem + "_other.json"), "r"))
        for ot in other:
            ot_id = ot["id"]
            # tg_fix = ot["target_fix"]
            # data[ot_id]["target"] = tg_fix + " | " + data[ot_id]["target"]
            data[ot_id]["target"] = ot["target"] 
        json.dump(data, open(file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    
    for file in eval_folder.rglob("*_other.json"):
        file.unlink()

if __name__ == "__main__":

    running_df = "/ltstorage/home/2pan/dataset/MMA_Anony/consolidated_running.csv"
    df = pd.read_csv(running_df)
    eval_folder = Path("evaluations/mma/no_am/")
    eval()
    # fix_other()
    # merge_other()
    # parse_other()


            

