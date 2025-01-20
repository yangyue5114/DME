1# --------------------------------------------------------
# Set-of-Mark (SoM) Prompting for Visual Grounding in GPT-4V
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by:
#   Jianwei Yang (jianwyan@microsoft.com)
#   Xueyan Zou (xueyan@cs.wisc.edu)
#   Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import argparse
import sys
sys.path.append('/mnt/petrelfs/yangyue/SoM')

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive

from scipy.ndimage import label
import numpy as np
import json
import matplotlib.pyplot as plt
import jsonlines
from tqdm import tqdm

from PIL import Image
import os
import jsonlines

'''
build args
'''

original_base_path = "/mnt/petrelfs/yangyue/SoM"
semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "/mnt/petrelfs/yangyue/SoM/swinl_only_sam_many2many.pth"
sam_ckpt = "/mnt/petrelfs/yangyue/SoM/sam_vit_h_4b8939.pth"
seem_ckpt = "/mnt/petrelfs/yangyue/SoM/seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(os.path.join(original_base_path, semsam_cfg))
opt_seem = load_opt_from_config_file(os.path.join(original_base_path,seem_cfg))
opt_seem = init_distributed_seem(opt_seem)


'''
build model
'''
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)


@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    _image = image.convert('RGB')
    #_mask = image['layers'][0].convert('L') if image['layers'] else None

    if slider < 1.5:
        model_name = 'seem'
    elif slider > 2.5:
        model_name = 'sam'
    else:
        if mode == 'Automatic':
            model_name = 'semantic-sam'
            if slider < 1.5 + 0.14:
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = 'sam'


    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if model_name == 'semantic-sam':
            model = model_semsam
            output, mask = inference_semsam_m2m_auto(model, _image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, *args, **kwargs)

        elif model_name == 'sam':
            model = model_sam
            if mode == "Automatic":
                output, mask = inference_sam_m2m_auto(model, _image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_sam_m2m_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        elif model_name == 'seem':
            model = model_seem
            if mode == "Automatic":
                output, mask = inference_seem_pano(model, _image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_seem_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)
        #print(output)
        return output, mask




def make_per_image_dir(image_path,task_dir):
    
    file_name_without_extension = os.path.basename(image_path).replace('.','_')
    
    per_img_dir = os.path.join(task_dir, file_name_without_extension)
    os.makedirs(per_img_dir, exist_ok=True)
    # print(per_img_dir)

    return per_img_dir



def adjust_and_infer(image, mode, slider_alpha, label_mode, anno_mode):
    # 定义初始参数和步长
    param = 1
    step = 0.1

    # output, output_mask = inference(image, 1, mode, slider_alpha, label_mode, anno_mode)
    cnt = 0
    while True:
        cnt+=1
        if cnt > 10:
            raise Exception("No enough masks")
        try:
            output, output_mask = inference(image, param, mode, slider_alpha, label_mode, anno_mode)
        except:
            param += step
            continue
        
        if len(output_mask) > 8:
            break
        param += step  # 增加参数值以进行下一次尝试

    return output, output_mask, param



def per_image_seg_and_mask(per_data, dataset_save_dir, error_list):
    
    if "index_path" in per_data:
        img_path = per_data["index_path"]
        #print("seed", img_path)
    elif "img_path" in per_data:
        img_path = per_data["img_path"]
    elif "image" in per_data:
        img_path = per_data["image"].replace("s3://coco_caption/val2014","/mnt/petrelfs/yangyue/prepare_rbt_dynamic_eval/exp_1_more_vlm_task/image_caption/coco_caption_100")
    else:
        assert 1==2
        
    # question = per_data["question"]
    
    image = Image.open(img_path)
    slider = 1 #gr.Slider(1, 3, value=2, label="Granularity", info="Choose in [1, 1.5), [1.5, 2.5), [2.5, 3] for [seem, semantic-sam (multi-level), sam]")
    mode = 'Automatic' #gr.Radio(['Automatic', 'Interactive', ], value='Automatic', label="Segmentation Mode")
    slider_alpha = 0.05 #调mask深浅 #gr.Slider(0, 1, value=0.1, label="Mask Alpha", info="Choose in [0, 1]")
    label_mode = 'Number' #gr.Radio(['Number', 'Alphabet'], value='Number', label="Mark Mode")
    anno_mode = ['Mask', 'Mark'] #gr.CheckboxGroup(choices=["Mask", "Box", "Mark"], value=['Mask', 'Mark'], label="Annotation Mode")
    
    
    try:
        output, output_mask, final_param = adjust_and_infer(image, mode, slider_alpha, label_mode, anno_mode)
        
        per_image_dir = make_per_image_dir(img_path, dataset_save_dir)
        mask_path_list = []
        for index,mask in enumerate(output_mask):
            segmentation = mask['segmentation']
            # 将布尔数组转换为0和1的整数数组
            mask_image = segmentation.astype(int)

            # 显示遮罩图像
            plt.imshow(mask_image, cmap='gray')
            plt.title('Mask Image')
            plt.axis('off')
            plt.show()

            # 保存为图像文件
            show_index = index+1
            per_mask_path = os.path.join(per_image_dir, str(show_index)+"-mask.jpg")
            # print(per_mask_path)
            mask_path_list.append(per_mask_path)
            plt.imsave(per_mask_path, mask_image, cmap='gray')

        per_image_som_path = os.path.join(per_image_dir, "som_result.jpg")
        img = Image.fromarray(output)
        img.save(per_image_som_path)
        
        per_data["new_per_image_som_path"] = per_image_som_path
        # per_data["new_per_image_mask_dir"] = per_image_dir
        per_data["mask_path_list"] = mask_path_list
        # per_data[]
        # per_data["per_question"] = question
        # per_image_gpt_query = {
        #     "per_image_som_path": per_image_som_path,
        #     "per_question": question
        # }
        # print(per_data)
        return per_data
    
    except Exception as e:
        error_list.append(img_path)
        return str(e), img_path

    

'''
launch app
'''



#MM
#keys:['attribute_comparison', 'image_style', 'social_relation', 'action_recognition', 'structuralized_imagetext_understanding', 'identity_reasoning', 'future_prediction', 'spatial_relationship', 'object_localization', 'function_reasoning', 'image_scene', 'physical_relation', 'image_topic', 'image_emotion', 'ocr', 'attribute_recognition', 'nature_relation', 'physical_property_reasoning', 'image_quality', 'celebrity_recognition']
#/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox/MM_bench/MM_bench_sample10_for_box.json
#add: ['attribute_comparison', 'social_relation', 'identity_reasoning', 'future_prediction', 'spatial_relationship', 'object_localization', 'function_reasoning', 'image_scene', 'physical_relation', 'image_emotion', 'ocr', 'attribute_recognition', 'nature_relation','celebrity_recognition']
#remove: 同add

#SEED
#/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox/SEED_Bench/SEED_Bench_sample10_for_box.json        
#dict_keys(['Visual_Reasoning', 'Instance_Identity', 'Instance_Interaction', 'Spatial_Relation', 'Scene_Understanding', 'Instance_Attributes', 'Instance_Location', 'Text_Understanding', 'Instances_Counting'])
#add: ['Visual_Reasoning', 'Instance_Identity', 'Instance_Interaction', 'Spatial_Relation', 'Scene_Understanding', 'Instance_Attributes', 'Instance_Location', 'Text_Understanding', 'Instances_Counting']
#remove: ['Visual_Reasoning', 'Instance_Identity', 'Scene_Understanding', 'Instance_Attributes', 'Text_Understanding', 'Instances_Counting']

#MME
#["celebrity", "color", "posters", "commonsense_reasoning", "scene", "position", "count", "landmark"] 
# ["celebrity", "color", "existence", "posters", "commonsense_reasoning", "scene", "position", "count", "landmark", "artwork"] 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args.')
    parser.add_argument('--data_path', type=str, default="/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox_gpt/SEED_Bench/SEED_Bench_sample10_for_box.json", help='root path to the dataset')
    parser.add_argument('--data_items', nargs='+', default=['Visual_Reasoning', 'Instance_Identity', 'Instance_Interaction', 'Spatial_Relation', 'Scene_Understanding', 'Instance_Attributes', 'Instance_Location', 'Text_Understanding', 'Instances_Counting'], help='data_items')
    parser.add_argument('--save_dir', type=str, default="/mnt/petrelfs/yangyue/SoM/som_remove/new/SEEDBench", help='save_dir')
    args = parser.parse_args()
    
    save_dir = args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    data_items = args.data_items
    print(data_items)

    image_gpt_query_list = []
    error_list = []
    data_dict = json.load(open(args.data_path))
    
    writer = jsonlines.open(f'{save_dir}/debug_image_gpt_query_info.jsonl', mode='w')
    error_writer = jsonlines.open(f'{save_dir}/detail_error_list_info.jsonl', mode='w')
    
    for data_item in data_items:
        task_dir = os.path.join(save_dir,data_item)
        os.makedirs(task_dir, exist_ok=True)
        # write_file_path = os.path.join(args.save_dir, data_item+'.jsonl')
        # writer = jsonlines.open(write_file_path, mode='w')
        # print(data_dict.keys())
        task_datas = data_dict[data_item]
        
        for per_data in tqdm(task_datas):
            per_image_gpt_query = per_image_seg_and_mask(per_data, task_dir, error_list)
            if isinstance(per_image_gpt_query, tuple):
                error_writer.write({data_item: per_image_gpt_query})
                error_writer._fp.flush()
            else:
                writer.write(per_image_gpt_query)
                writer._fp.flush()
            # if per_image_gpt_query!=False:
                #image_gpt_query_list.append(per_image_gpt_query)
    writer.close()
    error_writer.close()
    
    # with open(f'{save_dir}/image_gpt_query_info.json', "w") as f:
    #     json.dump(image_gpt_query_list, f, indent=4, ensure_ascii=False, separators=(',', ': '))
    
    # with open(f'{save_dir}/error_list_info.json', "w") as f:
    #     json.dump(error_list, f, indent=4, ensure_ascii=False, separators=(',', ': '))












    # def convert_to_serializable(obj):
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     elif isinstance(obj, list):
    #         return [convert_to_serializable(item) for item in obj]
    #     elif isinstance(obj, dict):
    #         return {key: convert_to_serializable(value) for key, value in obj.items()}
    #     else:
    #         return obj
    # serializable_list = convert_to_serializable(output_mask)
    # with open('/mnt/petrelfs/yangyue/SoM/examples/position/000000509699_mask.json', 'w') as json_file:
    #     json.dump(serializable_list, json_file, indent=4)