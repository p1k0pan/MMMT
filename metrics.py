import sacrebleu
# from sacrebleu.metrics import BLEU, CHRF, TER
from bert_score import score
import json
import sys
from nltk.translate import meteor_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import nltk
nltk.data.path.append('/mnt/data/users/liamding/data/LLAVA-2')
bert_name = "bert-base-chinese"
# bert_name = "/mnt/data/users/liamding/data/LLAVA-2/bert-base-chinese/"
tokenizer = AutoTokenizer.from_pretrained(bert_name, trust_remote_code=True)
print("load tokenizer")
def bleu_score(predict, answer):
    """
    refs = [ 
             ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'], 
           ]
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    """
    bleu = sacrebleu.corpus_bleu(predict, answer, lowercase=True, tokenize="flores101") #for chinese
    # bleu = sacrebleu.corpus_bleu(predict, answer, lowercase=True, tokenize="13a")
    return bleu.score

def chrf_score(predict, answer):
    chrf = sacrebleu.corpus_chrf(predict, answer)
    return chrf.score

def ter_score(predict, answer):
    ter = sacrebleu.corpus_ter(predict, answer, asian_support=True, normalized=True, no_punct=True)
    return ter.score

def bertscore(predict, answer):
    P, R, F1 = score(predict, answer, model_type = bert_name, device='cuda') #chinese
    # P, R, F1 = score(predict, answer, lang="de")
    return torch.mean(P).item(), torch.mean(R).item(), torch.mean(F1).item()

def meteor(predict, answer, type):
    # print(answer)
    # print(predict)
    # print(tokenizer.tokenize(predict[0]))
    # print(tokenizer.tokenize(answer[0]))
    all_meteor = []
    for i in range(len(predict)):
        meteor = meteor_score.meteor_score(
                            [tokenizer.tokenize(answer[i])],
                            tokenizer.tokenize(predict[i]),
                            )
        all_meteor.append(meteor)
    if type == "total":
        return sum(all_meteor) / len(all_meteor)
    else:
        return all_meteor[0]

def extract(data, target):
    predicts = []
    answers = []
    log_probs = []
    print("Extracting data")
    for item in tqdm(data):
        id = item["id"]
        pred = item['target'].strip()
        ans = target[id].strip()
        # lp = item['log_sentence_prob']
        lp = None
        predicts.append(pred)
        answers.append(ans)
        log_probs.append(lp)
    return predicts, answers, log_probs

def cal_total_metrics(predicts, answers):
    bs = bleu_score(predicts, [answers])
    cs = chrf_score(predicts, [answers])
    ts = ter_score(predicts, [answers])
    p, r, f1 = bertscore(predicts, answers)
    m = meteor(predicts, answers, "total")
    print("BLEU:", bs)
    print("CHRF:", cs)
    print("TER:", ts)
    print("BERT-P:", p, "BERT-R:", r, "BERT-F1:", f1)
    print("METEOR:", m)

    res = [{"BLEU": bs, "CHRF": cs, "TER": ts, "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "METEOR": m, "CHRF<10": chrf_10}]
    df = pd.DataFrame(res)
    df.to_csv(file.with_name(file.stem + "_total.csv"), index=False, encoding='utf-8-sig' )

def cal_each_metrics(predicts, answers):
    # print(predicts[0])
    # print(answers[0])
    all_result = []
    chrf_10 = 0
    for i in tqdm(range(len(predicts))):
        ans= answers[i]
        pred = predicts[i]
        bs = bleu_score([pred], [[ans]]) 
        cs = chrf_score([pred], [[ans]])
        if cs<10:
            chrf_10+=1
        ts = ter_score([pred], [[ans]])
        p, r, f1 = bertscore([pred], [ans])
        m = meteor([pred], [ans], "each")
        # print("BLEU:", bs)
        # print("CHRF:", cs)
        # print("TER:", ts)
        # print("BERT-P:", p, "BERT-R:", r, "BERT-F1:", f1)
        # print("METEOR:", m)
        all_result.append({"reference": ans, "predicts": pred, "BLEU": bs, "CHRF": cs, "TER": ts, 
                           "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "METEOR": m})
    df = pd.DataFrame(all_result)
    df.to_csv(file.with_name(file.stem + "_each.csv"), index=False, encoding='utf-8-sig')
    print("CHRF<10:", chrf_10)
    return chrf_10



if __name__ == "__main__":
    # data_file = "evaluations/3am/no_am_victx/normal"
    data_file = "evaluations/msctd/no_am/vlimcd_r"

    data_path = Path(data_file)
    # target_file = "/mnt/data/users/liamding/data/3AM/3AM/data/test.zh"
    # target_file = "/ltstorage/home/2pan/dataset/multi30k/data/task1/test/test_2016_flickr.de"
    target_file = "/mnt/data/users/liamding/data/MSCTD/MSCTD_data/enzh/chinese_test.txt"
    with open(target_file, "r", encoding="utf-8") as f:
        target = f.readlines()
    for file in data_path.rglob("*.json"):
        if os.path.exists(file.with_name(file.stem + "_total.csv")):
            continue
        print(file)
        data = json.load(open(file, "r", encoding="utf-8"))
        # bleu = BLEU()
        # ter = TER()
        predicts, answers, log_probs = extract(data, target)   
        # with open(file.with_name(file.stem + "_log_prob.txt"), "w", encoding="utf-8") as f:
        #     for lp in log_probs:
        #         f.write(str(lp) + "\n")
        print("each metrics")
        chrf_10 = cal_each_metrics(predicts, answers)
        print("total metrics")
        cal_total_metrics(predicts, answers)