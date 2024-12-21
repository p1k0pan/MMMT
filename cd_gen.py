import argparse
from email.mime import image
import torch
import sys
import copy

from llava_icd.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava_icd.conversation import conv_templates, SeparatorStyle
from llava_icd.model.builder import load_pretrained_model
from llava_icd.utils import disable_torch_init
from llava_icd.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava_icd.vcd_add_noise import add_diffusion_noise 

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import json
import tqdm
from tqdm.contrib import tzip
from pathlib import Path
import random
# def image_parser(args):
#     out = args.image_file.split(args.sep)
#     return out
def image_parser(image_file):
    out = image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def process_query(qs):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def generate(text, image_file, prompt_temp, tg):
    qs = prompt_temp + "\n"+ text
    prompt = process_query(qs)

    image_files = image_parser(image_file)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    attention_mask = None

    if "v" in cd: #"vicd", "vcd", "vlicd"
        images_tensor_cd = add_diffusion_noise(images_tensor, 500)
    else:
        images_tensor_cd = None

    if "i" in cd: #"icd", "vicd", "vlicd":
        # language cd
        cd_sp = "You are a confused object detector and you will ignore every details in the image."
    else:
        cd_sp = None
    
    if "l" in cd: #"vlicd","licd", "lcd":
        cd_prompt_temp = "Please translate the following English sentence into German:"
        qs = cd_prompt_temp + "\n"+ text
        prompt_lcd = process_query(qs)
    else:
        prompt_lcd = None
    
    if "s" in cd:
        sid=True
    else:
        sid=False

    if cd_sp is not None:
        prompt_icd = copy.copy(prompt)
        prompt_icd = cd_sp + " USER:" + prompt_icd.split("USER:")[1]
    
        input_ids_icd = (
            tokenizer_image_token(prompt_icd, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        # attention_mask_icd = input_ids_icd.ne(tokenizer.pad_token_id).long()
    else:
        input_ids_icd = None
        # attention_mask_icd = None

    if prompt_lcd is not None:
        input_ids_lcd = (
            tokenizer_image_token(prompt_lcd, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        # attention_mask_lcd = input_ids_lcd.ne(tokenizer.pad_token_id).long()
    else:
        input_ids_lcd = None
        # attention_mask_lcd = None
    
    if "m" in cd:
        import torch.nn.functional as F
        input_ids_only_text = (
            tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        ) #纯翻译问题，没有template
        only_text_embeds, image_feat, num_col = model.prepare_ti_embeds(input_ids_only_text, images_tensor, attention_mask, image_sizes)
        image_features = image_feat / image_feat.norm(dim=1, keepdim=True)
        text_features = only_text_embeds / only_text_embeds.norm(dim=1, keepdim=True)
        similarity_matrix = image_features @ text_features.t()
        importance_scores = similarity_matrix.mean(dim=1)  # [N_i]
        # topk = int(image_feat.shape[0]*0.1)  # 取前10个最相关的图像token
        # if "ir" in cd:
        #     topk_indices = importance_scores.topk(topk, largest=False).indices
        # else:
        #     topk_indices = importance_scores.topk(topk, largest=True).indices
        topk_r = int(image_feat.shape[0]*pr)
        topk_ir = int(image_feat.shape[0]*(0.1-pr))
        topk_indices_r = importance_scores.topk(topk_r, largest=True).indices
        topk_indices_ir = importance_scores.topk(topk_ir, largest=False).indices
        topk_indices = torch.cat((topk_indices_r, topk_indices_ir), dim=-1)
        for i in range(len(topk_indices)):
            topk_indices[i] += topk_indices[i] // num_col
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    else:
        topk_indices = None
    
    if log_prob:
        target_input_ids = tokenizer(tg, return_tensors='pt')['input_ids'] 
        model.set_target_ids(target_input_ids)
    else:
        model.set_target_ids(None)

    with torch.inference_mode():
        output_ids, log_sentence_prob = model.generate(
            input_ids,
            attention_mask = attention_mask,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            output_attentions=True, #don't use sdpa attention
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            input_ids_icd = input_ids_icd,
            images_cd = images_tensor_cd,
            input_ids_lcd = input_ids_lcd,
            sid=sid,
            img_topk = topk_indices
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs, log_sentence_prob


def eval_model():
    results = []
    
    for id, ti in enumerate(tzip(source, img_source, target)):
        text = ti[0].strip()
        img = ti[1].strip()
        tg = ti[2].strip()

        outputs, log_sentence_prob = generate(text, image_folder+img, prompt_temp, tg)
        results.append({"id":id, "source": text, "target": outputs, "log_sentence_prob": log_sentence_prob})

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
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=False)
    parser.add_argument("--query", type=str, required=False)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
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

    # args.image_file = "/ltstorage/home/2pan/LLaVA/llava/eval/bird.jpeg"
    # args.query = "What is the color of the bird?"
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map="auto"
    )

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

    # prompt_temp = "请将下面的英文句子翻译成中文："
    prompt_temp = args.prompt_temp
    print(prompt_temp)
    # output_path = f"evaluations/3am/no_am/{prompt_temp}/"

    log_prob = False
    # icd, lcd, licd, vcd, vicd, vlicd, sd, sicd, slicd, vlcd
    # cds = ["", "icd","lcd", "licd", "vcd", "vicd", "vlicd", "mcd_r", "mcd_ir"] 
    # cds = ["", "icd","lcd", "licd"] 
    # cds = ["imcd_r", "limcd_r","imcd_ir", "limcd_ir", "vlimcd_r", "vlimcd_ir"]
    # cds = ["mcd_r", "imcd_r", "limcd_r", "vlimcd_r"]
    # cds = ["mcd_r", "imcd_r"]
    cds = ["limcd_r", "vlimcd_r"]
    per_relevents = [0.1, 0.08, 0.06, 0.05, 0.04, 0.02, 0]
    for pr in per_relevents:
        for cd in cds:
            output_path = f"{args.output_path}{cd}/"
            Path(output_path).mkdir(parents=True, exist_ok=True)
            # cd = cd+f"_{str(pr).replace('.','')}"
            if cd !="":
                output_name = f"{str(pr).replace('.','')}.json"
            else:
                output_name = "original.json"
            print(cd)
            eval_model()

