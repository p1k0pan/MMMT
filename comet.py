from comet import download_model, load_from_checkpoint
from pathlib import Path
import json
import pandas as pd
import os

# Choose your model from Hugging Face Hub
# model_path = download_model("Unbabel/XCOMET-XXL")
# or for example:
model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
model = load_from_checkpoint(model_path)

def metrics(data):
    model_output = model.predict(data, batch_size=8, gpus=1)
    score = model_output.scores
    sys_score= model_output.system_score
    i=0
    for item in data:
        item["score"] = score[i]
        item["id"]=i
        i+=1
    json.dump(data, open(file.with_name(file.stem + "_comet.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    df = pd.read_csv(file.with_name(file.stem + "_total.csv"))
    df["COMET"] = sys_score
    df.to_csv(file.with_name(file.stem + "_total.csv"), index=False, encoding='utf-8-sig' )


if __name__ == "__main__":
    target_file = "/ltstorage/home/2pan/dataset/3AM/data/test.zh"
    mt_folder = "/ltstorage/home/2pan/MMMT/evaluations/3am/no_am-outsource/"
    mt_folder = Path(mt_folder)
    with open(target_file, "r", encoding="utf-8") as f:
        target = f.readlines()

    for file in mt_folder.rglob("*.json"):
        if os.path.exists(file.with_name(file.stem + "_comet.json")) or file.stem.endswith("_comet"):
            continue
        print("processing", file)
        data = []
        mt_file = json.load(open(file, "r", encoding="utf-8"))
        for item in mt_file:
            data.append({"src": item["source"], "mt": item["target"], "ref": target[int(item["id"])]})

        print(len(data))
        metrics(data)