'''
Adapted from
https://github.com/thunlp/Migician/blob/main/eval/MIG_bench_eval.py # noqa
'''
import torch
import warnings
import torch
import json
import datetime
import random
import copy
from tqdm import tqdm
import torch.multiprocessing as mp
from queue import Empty
from PIL import Image
import string
from decord import VideoReader, cpu 
import warnings
import argparse
import logging
from utils import *
# Required for MedSG
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, LlamaTokenizer
import re
from transformers.image_utils import load_image
import os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())
### required by LLaVA-OneVision ###
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle
except:
    print('[WARNING] LLaVA hasn\'t been installed')

from modelscope import AutoTokenizer, AutoConfig, AutoModel

warnings.filterwarnings("ignore")


# Please note that: Due to the conflicts of required transformers versions by different MLLMs, you may need to set the correct version for your target.

PROMPT_TEMPLATE = {
    'format_qwen': 'Format: <|box_start|>(x1,y1),(x2,y2)<|box_start|>. Strictly follow this format. Output only the coordinates. No additional text or explanation.', 
    'format_qwen_2_5': 'Output the bounding box as <|box_start|>(x1,y1),(x2,y2)<|box_end|>.',
    'format_internvl2': 'Format:<box>[[x1,y1,x2,y2]]</box>. Don\'t generate addtional words.',
    'format_minicpm': 'Format:<box>x1 y1 x2 y2</box>. Don\'t generate addtional words.',
    'format_mantis': ' Format:<box>[x1, y1, x2, y2]</box>. Don\'t generate addtional words.',
    'format_llava': ' Format: <box>[x1, y1, x2, y2]</box>, where x1~x2 and y1~y2 are real numbers in the range [0, 1] (normalized coordinates). Do not generate any additional words.'
}

def compute_iou(ground_truth, prediction, acc, task):
    iou = calculate_iou(ground_truth, prediction)
    if not acc.get(task):
        acc[task]={'IOU@0.7':0,'IOU@0.5':0,'IOU@0.3':0,'AVE_IOU':0,'TOTAL':0}
    if iou >= 0.7: acc[task]['IOU@0.7'] += 1
    if iou >= 0.5: acc[task]['IOU@0.5'] += 1
    if iou >= 0.3: acc[task]['IOU@0.3'] += 1
    acc[task]['AVE_IOU'] += iou
    acc[task]['TOTAL'] += 1
    return iou, acc

def post_processing(acc, output_path, output):
    for task, obj in acc.items():
        print(f"✅ Results for [{task}]:")
        print(obj)
        for item, value in obj.items():
            if item != 'TOTAL':
                print(f"|——> {item}:{100*value/acc[task]['TOTAL']}%  ✨")
    output.append(acc)
    with open(output_path,'w') as file2:
        json.dump(output, file2, indent=4, ensure_ascii=False)

################# Input process functions for specific MLLMs #################

def qwen2_process(obj):
    messages = [{"role": "user","content": []}]
    for path in obj['images']:
        messages[0]["content"].append({"type": "image","image": path})
    
    obj['question'] += PROMPT_TEMPLATE['format_qwen']
    messages[0]["content"].append({"type": "text", "text": obj['question']})
    return messages

def qwen2_respond(model, processor, messages, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,return_tensors="pt")
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response

def llava_process(obj, config, tokenizer, image_processor, device):
    conv_template = "qwen_1_5"
    images = []
    prefix = ''
    for idx, image_path in enumerate(obj['images']):
        image = Image.open(image_path)
        prefix += f'image{idx+1}:<image>\n'
        images.append(image)
    image_tensor = process_images(images, image_processor, config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    obj['question'] = prefix + obj['question'].replace('<image>','') + PROMPT_TEMPLATE['format_mantis']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    question = re.sub(r'\(<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>\)', '', obj['question']) # [0.\1,0.\2,0.\3,0.\4]
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    # print(prompt_question)
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size for image in images]

    if obj['task'] == 'multi_view':
        width, height = obj['size']
    else:
        width, height = 336,336
    
    return input_ids, image_tensor, image_sizes, width, height

def mplug_process(obj, config, tokenizer, processor, device):
    prefix, images = '', []
    for idx, image_path in enumerate(obj['images']):
        image = Image.open(image_path).convert('RGB')
        prefix += f'Image-{idx+1}:<|image|>\n'
        images.append(image)
    
    obj['question'] = prefix + obj['question'].replace('<image>','') + PROMPT_TEMPLATE['format_mantis']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    question = re.sub(r'\(<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>\)', '', obj['question']) # [0.\1,0.\2,0.\3,0.\4]
    
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""}
    ]
    inputs = processor(messages, images=images, videos=None).to(device)
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens': 256,
        'decode_text': True,
    })

    if obj['task'] == 'multi_view':
        width, height = obj['size']
    else:
        width, height = 336,336

    return inputs, width, height

def internvl2_process(obj):
    prefix, num_patches_list, first = '', [], True
    for idx, image_path in enumerate(obj['images']):
        temp, ori_width, ori_height = load_image_intenvl2(image_path, max_num=12)
        temp = temp.to(torch.bfloat16).cuda()
        if idx == 0:
            width_1, height_1 = ori_width, ori_height
            
        num_patches_list.append(temp.size(0))
        prefix += f'Image-{idx+1}: <image>\n'
        if first == True:
            pixel_values = temp
        else:
            pixel_values = torch.cat((pixel_values, temp), dim=0)
        first = False
    
    obj['question'] = prefix + obj['question'].replace('<image>','') + PROMPT_TEMPLATE['format_internvl2']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    # question = re.sub(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', r'[\1,\2,\3,\4]', obj['question'])

    question = obj['question']
    matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', question)

    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        x1n = round(x1 / width_1 * 1000)
        y1n = round(y1 / height_1 * 1000)
        x2n = round(x2 / width_1 * 1000)
        y2n = round(y2 / height_1 * 1000)

        orig = f"({x1},{y1}),({x2},{y2})"
        new = f"[{x1n},{y1n},{x2n},{y2n}]"
        question = question.replace(orig, new)

    if obj['task'] == 'multi_view':
        width, height = obj['size']
    else:
        width, height = 336,336
    return pixel_values, question, num_patches_list, width, height

def mantis_process(obj):
    images = []
    messages = [{"role": "user","content": []}]
    for idx, image_path in enumerate(obj['images']):
        image = load_image(image_path)

        if idx == 0:
            width_1, height_1 = image.size

        images.append(image)
        messages[0]['content'].append({"type": "image"})
        
    obj['question'] += PROMPT_TEMPLATE['format_mantis']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    # question = re.sub(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', r'[0.\1,0.\2,0.\3,0.\4]', obj['question'])
    question = obj['question']
    matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', question)

    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        x1n = round(x1 / width_1, 4)
        y1n = round(y1 / height_1, 4)
        x2n = round(x2 / width_1, 4)
        y2n = round(y2 / height_1, 4)

        orig = f"({x1},{y1}),({x2},{y2})"
        new = f"[{x1n},{y1n},{x2n},{y2n}]"
        question = question.replace(orig, new)

    messages[0]['content'].append({"type": "text","text":question})
    if obj['task'] == 'multi_view':
        width, height = obj['size']
    else:
        width, height = 336,336
    return messages, images, width, height

def migician_process(obj):
    messages = [{"role": "user","content": []}]
    for idx, path in enumerate(obj['images']):
        image = load_image(path)

        if idx == 0:
            width_1, height_1 = image.size
        messages[0]["content"].append({"type": "image","image": path})
    
    matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', obj['question'])
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        x1n = round(x1 / width_1 * 1000)
        y1n = round(y1 / height_1 * 1000)
        x2n = round(x2 / width_1 * 1000)
        y2n = round(y2 / height_1 * 1000)

        orig = f"({x1},{y1}),({x2},{y2})"
        new = f"[{x1n},{y1n},{x2n},{y2n}]"
        obj['question'] = obj['question'].replace(orig, new)
    obj['question'] += PROMPT_TEMPLATE['format_qwen']
    messages[0]["content"].append({"type": "text", "text": obj['question']})
    return messages

def migician_respond(model, processor, messages, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,return_tensors="pt")
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response

def minicpm_process(obj):
    content = []
    for idx, image_path in enumerate(obj['images']):
        temp = Image.open(image_path).convert('RGB')
        if idx == 0:
            width_1, height_1 = temp.size
        temp = resize_image(temp)
        content.append(temp)
        
    obj['question'] += PROMPT_TEMPLATE['format_minicpm']
    obj['question'] = obj['question'].replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>')
    # question = re.sub(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', r'[\1,\2,\3,\4]', obj['question'])

    question = obj['question']
    matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', question)

    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        x1n = round(x1 / width_1 * 1000)
        y1n = round(y1 / height_1 * 1000)
        x2n = round(x2 / width_1 * 1000)
        y2n = round(y2 / height_1 * 1000)

        orig = f"({x1},{y1}),({x2},{y2})"
        new = f"[{x1n},{y1n},{x2n},{y2n}]"
        question = question.replace(orig, new)

    content.append(question)

    if obj['task'] == 'multi_view':
        width, height = obj['size']
    else:
        width, height = 336,336
    return [{'role': 'user', 'content': content}], width, height


################# Model responding functions for specific MLLMs #################
def qwen2_vl_eval(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"qwen2_vl_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages = migician_process(obj)
        response = migician_respond(model, processor, messages, device)
        if obj['task'] == 'multi_view':
            width, height = obj['size']
        else:
            width, height = 336,336
        print(response)
        prediction = extract_bbox_old(response[0], width, height) 
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def qwen2_5_vl_eval(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"qwen2_5_vl_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages = qwen2_process(obj)
        response = qwen2_respond(model, processor, messages, device)
        print(response)
        prediction = extract_bbox_strong(response[0])
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def MedSG_eval(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"MedSG_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages = qwen2_process(obj)
        response = qwen2_respond(model, processor, messages, device)
        print(response)
        prediction = extract_bbox_strong(response[0])
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def llava_eval(model, tokenizer, image_processor, test_data, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"llava-onevision_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}
    
    model = model.to(device)  # remove when using multi-GPU

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        input_ids, image_tensor, image_sizes, width, height = llava_process(obj, model.config, tokenizer, image_processor, device)
        cont = model.generate(input_ids, images=image_tensor, image_sizes=image_sizes, do_sample=False, temperature=0, max_new_tokens=100)
        response = tokenizer.batch_decode(cont, skip_special_tokens=True)
        
        prediction = extract_bbox_old(response[0], width=width, height=height)
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def mplug_eval(model, tokenizer, config, processor, test_data, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"mplug-owl3_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        inputs, width, height = mplug_process(obj, config, tokenizer, processor, device)
        response = model.generate(**inputs)
        prediction = extract_bbox_old(response[0], width=width, height=height) 
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def internvl2_eval(model, tokenizer, generation_config, test_data, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"internvl2_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    model.to(device)  

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        pixel_values, question, num_patches_list, width, height = internvl2_process(obj)
        pixel_values = pixel_values.to(device)  
        response,_ = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        
        prediction = extract_bbox_old(response,width=width,height=height) 
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        if response is None:
            answer = ''
        else:
            answer = response[0]
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': answer, 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)

def mantis_eval(model, generation_kwargs, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"mantis_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages, images, width, height = mantis_process(obj)
        messages = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=messages, images=images, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, **generation_kwargs)
        response = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        prediction = extract_bbox_old(response[0], width=width, height=height) 
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def minicpm_eval(model, tokenizer, test_data, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"minicpm_{test_data[0]['task']}_{current_time}.json")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)[:-1]
    output, acc = [], {}

    model.to(device)

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        msgs, width, height = minicpm_process(obj)
        response = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        prediction = extract_bbox_old(response,width=width,height=height)
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0], 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)


def model_selection(model_type, model_path, test_data, device, output_path):
    if model_type=='MedSG':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        MedSG_eval(model, test_data, processor, device, output_path)
    elif model_type=='migician':
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        qwen2_vl_eval(model, test_data, processor, device, output_path)
    elif model_type=='qwen2_5_vl':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        qwen2_5_vl_eval(model, test_data, processor, device, output_path)
    elif model_type=='llava_onevision':
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, "llava_qwen", device_map={"": 4})
        model.eval()
        llava_eval(model, tokenizer, image_processor, test_data, device, output_path)    
    elif model_type=='mplug_owl3':
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = model.init_processor(tokenizer)
        mplug_eval(model, tokenizer, config, processor, test_data, device, output_path)
    elif model_type=='internvl2':
        # device_map = split_model_3(model_path)
        # device_map = split_model_2_5(model_path)
        # device_map = split_model_2(model_path)
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        internvl2_eval(model, tokenizer, generation_config, test_data, device, output_path)
    elif model_type=='mantis':
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
        generation_kwargs = {"max_new_tokens": 1024, "num_beams": 5, "do_sample": True}
        mantis_eval(model, generation_kwargs, test_data, processor, device, output_path)
    elif model_type=='minicpm':
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        model = model.eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        minicpm_eval(model, tokenizer, test_data, device, output_path)
    else:
        raise NotImplementedError
    
########################### Task-specific Calling ###########################

def Registered_diff(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'Registered_Diff']
    model_selection(model_type, model_path, test_data, device, output_path)

def Non_Registered_diff(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'Non_Registered_Diff']
    model_selection(model_type, model_path, test_data, device, output_path)

def multi_view(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'multi_view']
    model_selection(model_type, model_path, test_data, device, output_path)

def object_tracking(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'object_tracking']
    model_selection(model_type, model_path, test_data, device, output_path)   

def concept(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'concept']
    model_selection(model_type, model_path, test_data, device, output_path) 

def patch(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'patch']
    model_selection(model_type, model_path, test_data, device, output_path) 

def crossmodal(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'crossmodal']
    model_selection(model_type, model_path, test_data, device, output_path) 

def referring(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'referring']
    model_selection(model_type, model_path, test_data, device, output_path) 

def parse_args():
    parser = argparse.ArgumentParser(description="Run MedSG evaluation.")

    parser.add_argument("--model_type", type=str, default='qwen2_5_vl', help="Model type, e.g., internvl2, qwen2_vl, etc.")
    parser.add_argument("--model_path", type=str, default='/your/checkpoint', help="Path to model checkpoint.")
    parser.add_argument("--task", type=str, required=True, 
                        choices=[
                            "Registered_Diff", "Non_Registered_Diff", "multi_view", "object_tracking", 
                            "concept", "patch", "crossmodal", "referring" 
                        ],
                        help="Evaluation task to run.")
    parser.add_argument("--test_data", type=str, default="/your/json_file/path", help="Path to test data JSON file.")
    parser.add_argument("--output_path", type=str, default="/your/output/path", help="Output path for evaluation results.")
    
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    with open(args.test_data) as file:
        test_data = json.load(file)

    os.makedirs(args.output_path, exist_ok=True)
    
    device = torch.device('cuda')
    if args.task == "Registered_Diff":
        Registered_diff(args.model_type, args.model_path, test_data, device, args.output_path)
    elif args.task == "Non_Registered_Diff":
        Non_Registered_diff(args.model_type, args.model_path, test_data, device, args.output_path)
    elif args.task == "multi_view":
        multi_view(args.model_type, args.model_path, test_data, device, args.output_path)
    elif args.task == "object_tracking":
        object_tracking(args.model_type, args.model_path, test_data, device, args.output_path)
    elif args.task == "concept":
        concept(args.model_type, args.model_path, test_data, device, args.output_path)
    elif args.task == "patch":
        patch(args.model_type, args.model_path, test_data, device, args.output_path)
    elif args.task == "crossmodal":
        crossmodal(args.model_type, args.model_path, test_data, device, args.output_path)
    elif args.task == "referring":
        referring(args.model_type, args.model_path, test_data, device, args.output_path)
    else:
        raise ValueError(f"Task {args.task} not supported.")