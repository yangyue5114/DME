import os
import requests
import jsonlines
import multiprocessing
from openai import OpenAI
from PIL import Image
import base64
import datetime
import json
import argparse
from tqdm import tqdm
import os
import json
import random
from math import ceil


def timestr(second=True):
    # 格式化当前时间字符串
    format_str = '%m%d-%H-%M-%S' if second else '%m%d-%H-%M'
    return datetime.now().strftime(format_str)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_shape(image_path):
    img = Image.open(image_path)  # 替换为你的图片文件路径
    width, height = img.size
    return [width, height]


client = OpenAI(
    base_url='https://api.openai-sb.com/v1',
    api_key='xxx',
)


text_input = '''Based on the image I provided (image includes the mark and the mask for each object), tell me which objects can be removed without changing the answer to the question: '{}', 
                Please give me a list containing exactly 5 objects can be removed. Do not output any extra information! Just output a list! The example of output format is:
                [
                    {{"object_mark": xxx, "object_name": xxx}},
                    {{"object_mark": xxx, "object_name": xxx}},
                    ...
                ]
                where "object_mark" is the given mark of object, and "object_name" is the name of specific object(can be repeated without distinction). Note that the removed object can not change the answer to the question! In the order you think is most obvious and appropriate to remove.
                '''
                


def process_file(item):
    # print("---",item)
    # try:
    img_path = item["new_per_image_som_path"]
    print(img_path)
    question = item["question"]
    print("---",question)
    base64_image = encode_image(img_path)
    #print('---',base64_image)
    input_text = text_input.format(question)
    
    # print('---',input_text)
    
    # print(input_text)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": input_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    content = response.choices[0].message.content
    return json.dumps({"per_image_som_path":img_path, "per_question":question, "gpt_response": content}, ensure_ascii=False), True

    # except Exception as e:
    #     print("---",e)
    #     return json.dumps(img_path, ensure_ascii=False), False



# 多进程处理
def main(img_dicts, writer):
    # with multiprocessing.Pool(processes=64) as pool:
    #     results = pool.map(partial(process_file), png_files)
    item_num_success = 0

    for img_item in tqdm(img_dicts):
        result, excep_flag = process_file(img_item)
        writer.write(result)
        writer._fp.flush()

        if excep_flag == True:
            item_num_success= item_num_success + 1
        
    
    print("item_num_success:",item_num_success)


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


# 使用，构建数据集
    # directory = '/mnt/petrelfs/yangyue/evaluation_datasets/SEED_Bench/category_jsons'  # 替换为你的目录路径
    # list_and_process_json_files(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args.')
    parser.add_argument('--json_path', type=str, default="/mnt/petrelfs/yangyue/SoM/som_remove/new/SEEDBench/image_gpt_query_info.json", help='root path to the dataset')
    # parser.add_argument('--save_dir', type=str, default="/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox/MM_bench/remove", help='save_dir')
    args = parser.parse_args()
    
    json_path = args.json_path
    # save_dir = args.save_dir
    # os.makedirs(args.save_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
        write_file_path = json_path.replace("image_gpt_query_info", "image_gpt_query_info_select_5")
        writer = jsonlines.open(write_file_path, mode='w')
        
        main(data, writer)

    

    # parser = argparse.ArgumentParser(description='Args.')
    # # parser.add_argument('--data_path', type=str, default="/mnt/petrelfs/yangyue/dynamic_eval/evaluation/MME_Benchmark_release_data.json", help='root path to the dataset')
    # # parser.add_argument('--data_items', nargs='+', default=["celebrity", "color", "posters", "commonsense_reasoning", "scene", "position", "count", "landmark"], help='data_items')
    # parser.add_argument('--save_dir', type=str, default="/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox/MM_bench/add", help='save_dir')
    # args = parser.parse_args()
    
    # os.makedirs(args.save_dir, exist_ok=True)
    # data_json_paths=[]

    # for data_json_path in data_json_paths:
    #     task_item = data_json_path.split('/')[-1].split('-')[0].replace(".json","")
    #     write_file_path = os.path.join(args.save_dir, task_item+'.jsonl')
    #     writer = jsonlines.open(write_file_path, mode='w')

    #     with open(data_json_path, 'r', encoding='utf-8') as file:
    #         img_dicts = json.load(file)
        
    #     main(img_dicts, writer)



