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
    base_url='https://api.openai.com/v1',
    api_key='xxx',
)


# image_input = "There is  a multi-choice question about the image (resolution:{}x{}) Its question is: \'{}\'. Its correct answer choice is: \'{}:{}\'. \n"
image_input = "There is a question about the image (resolution:{}x{}) Its question is: \'{}\'. Its correct answer choice is: \'{}\'. \n"

requirements = '''Now please add an object into this image, but keep the original answer of the question the same, which means the added object can not change the original answer and the position of the added object can not cover the visual answer information. Note that the size of the added object should be as large as possible, please audaciously magnify the size of the added object to differ from the original image.\n 
                Please give me randomly 10 objects can be added and give me the exact bounding box coordinates of each object according to the original resolution I give you. The box in the output is the coordinates of the upper left corner (x_min, y_min) and lower right corner (x_max, y_max) of the object. The output format should only be a list. 
                Each item in the list must be a dict.  Do not output any extra information! Just output a list! The example of output format is:\n
                [\n
                    {'name': name of object1, 'box':[x_min, y_min, x_max, y_max]},\n
                    {'name': name of object2, 'box':[x_min, y_min, x_max, y_max]}\n
                ]'''



def process_file(item):
    try:
        img_path = item["index_path"]
        img_shape = get_shape(img_path)
        
        base64_image = encode_image(img_path)
        choice = item["answer"]
        choice_answer = item[choice]
        # input_text = image_input.format(*img_shape, item["question"], choice, choice_answer) + requirements
        input_text = image_input.format(*img_shape, item["question"], choice_answer) + requirements
        
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
        return json.dumps({img_path: content}, ensure_ascii=False), True

    except Exception as e:
        print(e)
        return json.dumps(img_path, ensure_ascii=False), False



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
#add: ['attribute_comparison', 'social_relation', 'identity_reasoning', 'future_prediction', 'spatial_relationship', 'object_localization', 'function_reasoning', 'image_scene', 'physical_relation', 'image_emotion', 'ocr', 'attribute_recognition', 'nature_relation','celebrity_recognition']


#SEED       
#dict_keys(['Visual_Reasoning', 'Instance_Identity', 'Instance_Interaction', 'Spatial_Relation', 'Scene_Understanding', 'Instance_Attributes', 'Instance_Location', 'Text_Understanding', 'Instances_Counting'])
#add: ['Visual_Reasoning', 'Instance_Identity', 'Instance_Interaction', 'Spatial_Relation', 'Scene_Understanding', 'Instance_Attributes', 'Instance_Location', 'Text_Understanding', 'Instances_Counting']

#MME
#["celebrity", "color", "existence", "posters", "commonsense_reasoning", "scene", "position", "count", "landmark", "artwork"]        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args.')
    parser.add_argument('--data_path', type=str, default="/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox_gpt/MM_bench/MMBench_DEV_EN_sample30_for_box.json", help='root path to the dataset')
    parser.add_argument('--data_items', nargs='+', default=['attribute_comparison', 'image_style', 'social_relation', 'action_recognition', 'structuralized_imagetext_understanding', 'identity_reasoning', 'future_prediction', 'spatial_relationship', 'object_localization', 'function_reasoning', 'image_scene', 'physical_relation', 'image_topic', 'image_emotion', 'ocr', 'attribute_recognition', 'nature_relation', 'physical_property_reasoning', 'image_quality', 'celebrity_recognition'], help='data_items')
    parser.add_argument('--save_dir', type=str, default="/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox_new_generated_6_18/MM_bench/add", help='save_dir')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    data_items = args.data_items

    for data_item in data_items:
        write_file_path = os.path.join(args.save_dir, data_item+'.jsonl')
        writer = jsonlines.open(write_file_path, mode='w')
        data_dict = json.load(open(args.data_path))
        print(data_dict.keys())
        data = data_dict[data_item]
        main(data, writer)

    





