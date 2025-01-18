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
import jieba
from comet import download_model, load_from_checkpoint
def bleu_score(predict, answer, chinese=False):
    """
    refs = [ 
             ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'], 
           ]
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    """
    if chinese:
        # predict2 = [" ".join(jieba.cut(p, cut_all=False)) for p in predict]
        # answer2 = [[" ".join(jieba.cut(a, cut_all=False)) for a in answer[0]]]
        # bleu = sacrebleu.corpus_bleu(predict2, answer2, lowercase=True, tokenize="none")
        bleu = sacrebleu.corpus_bleu(predict, answer, lowercase=True, tokenize="zh")
    else:
        bleu = sacrebleu.corpus_bleu(predict, answer, lowercase=True, tokenize="13a")
    return bleu.score

def chrf_score(predict, answer):
    chrf = sacrebleu.corpus_chrf(predict, answer)
    return chrf.score

def chrfppp_score(predict, answer, chinese):
    if chinese:
        predict2 = [" ".join(jieba.cut(p, cut_all=False)) for p in predict]
        answer2 = [[" ".join(jieba.cut(a, cut_all=False)) for a in answer[0]]]
    else:
        predict2 = predict
        answer2 = answer
    
    chrfppp = sacrebleu.corpus_chrf(predict2, answer2, word_order=2)
    return chrfppp.score

def ter_score(predict, answer):
    ter = sacrebleu.corpus_ter(predict, answer, asian_support=True, normalized=True, no_punct=True)
    return ter.score

def bertscore(predict, answer, chinese=False):
    if chinese:
        P, R, F1 = score(predict, answer, model_type = bert_name, device="cuda")
    else:
        P, R, F1 = score(predict, answer, lang="de", device="cuda")
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
    sources = []
    log_probs = []
    comets = []
    print("Extracting data")
    for item in tqdm(data):
        id = item["id"]
        pred = item['target'].strip()
        ans = target[id].strip()
        src = item["source"].strip()
        comets.append({"src": src, "mt": pred, "ref": ans})
        # lp = item['log_sentence_prob']
        lp = None
        predicts.append(pred)
        answers.append(ans)
        sources.append(src)
        log_probs.append(lp)
    return predicts, answers, sources, comets ,log_probs

def cal_total_metrics(predicts, answers, chrf_10, comet_sys_score, chinese):
    bs = bleu_score(predicts, [answers], chinese=chinese)
    cs = chrf_score(predicts, [answers])
    cspp = chrfppp_score(predicts, [answers], chinese)
    ts = ter_score(predicts, [answers])
    p, r, f1 = bertscore(predicts, answers, chinese)
    m = meteor(predicts, answers, "total")
    print("BLEU:", bs)
    print("CHRF:", cs)
    print("TER:", ts)
    print("BERT-P:", p, "BERT-R:", r, "BERT-F1:", f1)
    print("METEOR:", m)
    print("COMET:", comet_sys_score)

    res = [{"BLEU": bs, "CHRF": cs, "CHRF++": cspp, "TER": ts, "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "METEOR": m, "CHRF<10": chrf_10, "COMET": comet_sys_score}]
    df = pd.DataFrame(res)
    df.to_csv(file.with_name(file.stem + "_total.csv"), index=False, encoding='utf-8-sig' )

def cal_each_metrics(predicts, answers, source, comets, chinese):
    # print(predicts[0])
    # print(answers[0])
    model_output = comet_model.predict(comets, batch_size=8, gpus=1)
    score = model_output.scores
    sys_score= model_output.system_score

    all_result = []
    chrf_10 = 0
    for i in tqdm(range(len(predicts))):
        ans= answers[i]
        pred = predicts[i]
        bs = bleu_score([pred], [[ans]], chinese=chinese) 
        cs = chrf_score([pred], [[ans]])
        cspp = chrfppp_score([pred], [[ans]], chinese)
        if cs<10:
            chrf_10+=1
        ts = ter_score([pred], [[ans]])
        p, r, f1 = bertscore([pred], [ans], chinese=chinese)
        m = meteor([pred], [ans], "each")
        # print("BLEU:", bs)
        # print("CHRF:", cs)
        # print("TER:", ts)
        # print("BERT-P:", p, "BERT-R:", r, "BERT-F1:", f1)
        # print("METEOR:", m)
        all_result.append({"reference": ans, "predicts": pred, "source":source[i], "BLEU": bs, "CHRF": cs, "CHRF++": cspp, "TER": ts, "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "METEOR": m, "COMET": score[i]})
    df = pd.DataFrame(all_result)
    df.to_csv(file.with_name(file.stem + "_each.csv"), index=False, encoding='utf-8-sig')
    print("CHRF<10:", chrf_10)
    return chrf_10, sys_score



if __name__ == "__main__":
    # data_file = "evaluations/msctd/no_am/mcd_r"
    data_file = "/ltstorage/home/2pan/MMMT/evaluations/3am/no_am-outsource/normal"
    data_path = Path(data_file)
    target_file = "/ltstorage/home/2pan/dataset/3AM/data/test.zh"
    # target_file = "/ltstorage/home/2pan/dataset/multi30k/data/task1/test/test_2016_flickr.de"
    # target_file = "/ltstorage/home/2pan/MSCTD_data/enzh/chinese_test.txt"
    source_file = "/ltstorage/home/2pan/dataset/3AM/data/test.en"
    with open(target_file, "r", encoding="utf-8") as f:
        target = f.readlines()
    with open(source_file, "r", encoding="utf-8") as f:
        source = f.readlines()
    
    chinese=True
    if chinese:
        bert_name = "bert-base-chinese"
    else:
        bert_name = "roberta-large" # for english
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    for file in data_path.rglob("*.json"):
        if os.path.exists(file.with_name(file.stem + "_total.csv")) or file.stem.endswith("_comet"):
            continue
        print(file)
        data = json.load(open(file, "r", encoding="utf-8"))
        # bleu = BLEU()
        # ter = TER()
        predicts, answers, sources, comets, log_probs = extract(data, target)   
        # with open(file.with_name(file.stem + "_log_prob.txt"), "w", encoding="utf-8") as f:
        #     for lp in log_probs:
        #         f.write(str(lp) + "\n")
        print("each metrics")
        chrf_10, comet_sys_score = cal_each_metrics(predicts, answers, sources ,comets, chinese)
        print("total metrics")
        cal_total_metrics(predicts, answers, chrf_10, comet_sys_score, chinese)