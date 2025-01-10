import argparse
import torch
import sys

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from llava_icd.vcd_add_noise import add_diffusion_noise 

from PIL import Image

import requests
import re
import json
import tqdm
from tqdm.contrib import tzip
from pathlib import Path
import random
def process_query(qs, image_file, sp=None):
    if sp is not None:
       messages = [
            {"role": "system", "content": sp},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": qs},
                ],
            }
        ] 
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": qs},
                ],
            }
        ]
    # Preparation for inference
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return prompt, image_inputs, video_inputs

def generate(text, image_file, prompt_temp):
    # cd_sp = "You are a confused object detector and you will ignore every details in the image."
    # cd_sp = "You are a professional translator with expertise in translating texts between Chinese and English. Your task is to provide accurate, fluent, and context-appropriate translations of the given text. When translating ambiguous words or phrases, rely on additional contextual clues, such as any accompanying images or visual references, to resolve the ambiguity and ensure precise meaning."
    cd_sp = "You are an amateur translator with limited experience in translating texts between Chinese and English. Your task is to provide rough, literal translations of the given text. Focus on word-for-word translation without worrying too much about fluency, accuracy, or context. Avoid interpreting ambiguous phrases and stick to a straightforward rendition, even if the result sounds awkward or unnatural."
    qs = prompt_temp + "\n"+ text
    prompt, image_inputs, video_inputs = process_query(qs, image_file, sp=cd_sp)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        # images=None,
        # videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    return output_text


def eval_model():
    results = []
    
    for id, ti in enumerate(tzip(source, img_source, target)):
        text = ti[0].strip()
        img = ti[1].strip()
        # tg = ti[2].strip()

        outputs = generate(text, image_folder+img, prompt_temp)
        results.append({"id":id, "source": text, "target": outputs, "log_sentence_prob": 0})

    json.dump(results, open(output_path + output_name, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

def shuffle_without_fixed_positions(img_source):
    indices = list(range(len(img_source)))
    while True:
        random.shuffle(indices)
        # 检查是否所有元素都没有留在原位置
        if all(i != indices[i] for i in range(len(indices))):
            break
    shuffled_imgs = [img_source[i] for i in indices]
    return shuffled_imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=False)
    # parser.add_argument("--query", type=str, required=False)
    # parser.add_argument("--conv-mode", type=str, default=None)
    # parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--source_file", type=str, required=True)
    parser.add_argument("--target_file", type=str, required=True)
    parser.add_argument("--image_source", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--prompt_temp", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()


    source_file = args.source_file
    with open(source_file, "r", encoding="utf-8") as f:
        source = f.readlines()
    
    target_file = args.target_file
    with open(target_file, "r", encoding="utf-8") as f:
        target = f.readlines()
    
    image_folder = args.image_folder
    image_source = args.image_source
    with open(image_source, "r", encoding="utf-8") as f:
        img_source = f.readlines()

    prompt_temp = args.prompt_temp
    # output_path = f"evaluations/3am/no_am/{prompt_temp}/"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )
    print(model)

    # default processer
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    log_prob = False
    # # icd, lcd, licd, vcd, vicd, vlicd, sd, sicd, slicd, vlcd
    # # cds = ["", "icd","lcd", "licd", "vcd", "vicd", "vlicd"] 
    # # cds = ["", "icd","lcd", "licd"] 
    # # cds = ["vcd", "vicd", "vlicd"]
    # # cds = ["imcd_r", "limcd_r","imcd_ir", "limcd_ir", "vlimcd_r", "vlimcd_ir"]
    # # cds = ["mcd_r", "imcd_r", "limcd_r", "vlimcd_r"]
    # cds = ["mcd_r", "imcd_r"]
    # # cds = ["limcd_r", "vlimcd_r"]
    cds = ["disturbance_ameture"]
    # per_relevents = [0.08, 0.06, 0.05, 0.04, 0.02, 0]
    per_relevents = [1]
    for pr in per_relevents:
        for cd in cds:
            print(cd)
            # output_path = f"{args.output_path}{cd}/"
            output_path = f"{args.output_path}normal/"
            Path(output_path).mkdir(parents=True, exist_ok=True)
            # cd = cd+f"_{str(pr).replace('.','')}"
            if cd !="":
                output_name = f"{cd}.json"
                # output_name = f"{str(pr).replace('.','')}.json"
            else:
                output_name = "original.json"
            print(output_name)
            eval_model()

    
