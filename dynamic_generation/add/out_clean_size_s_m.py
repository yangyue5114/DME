import json
import jsonlines
import argparse
import os
import re
import time



def per_json_box(per_box_json, save_dir):
    item_name = os.path.basename(per_box_json).split('.')[0]
    save_path = os.path.join(save_dir, item_name)
    os.makedirs(save_path, exist_ok=True)

    save_file_name = os.path.join(save_path, 'success.json')
    error_file_name = os.path.join(save_path, 'error.json')

    with jsonlines.open(per_box_json, mode='r') as reader:
        o_data = list(reader)

    error_files = []
    save_files = []
    for each_data in o_data:
        # 确认 each_data 的类型是字符串
        if isinstance(each_data, str):
            try:
                each_one = json.loads(each_data)
                # print("right:",each_one)
                if not isinstance(each_one, dict):
                    # print("false:",each_one)
                    print(f"Parsed data is not a dictionary, gpt4v no response: {each_one}")
                    error_files.append(each_data)
                    # raise ValueError('Parsed data is not a dictionary')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                error_files.append(each_data)
                continue
        elif isinstance(each_data, dict):
            each_one = each_data
        else:
            print(f"Unexpected data type: {type(each_data)}")
            error_files.append(each_data)
            continue
        
        print("each_one",each_one)
        if hasattr(each_one, 'keys'):
            name = list(each_one.keys())[0]
            print("name:",name)
            try:
                bboxlist = []
                value = each_one[name]  # list(each_one.values())[0]
                # print("value:",value)
                matches = re.findall(pattern, value)
                # print("matches:",matches)
                for match in matches:
                    try:
                        bbox_one = {}
                        bbox_one["name"] = match[0]
                        bbox_one["box"] = [int(match[1]), int(match[2]), int(match[3]), int(match[4])]
                        bboxlist.append(bbox_one)
                    except Exception as e:
                        print(f"Error processing match {match}: {e}")
                        continue
                if len(bboxlist) == 0:
                    raise ValueError('bboxlist is empty.')

            except Exception as e:
                print(f"Error processing {name}: {e}")
                error_files.append(name)
                continue
            save_files.append({'img_path': name, 'bboxes': bboxlist})

    
    with open(save_file_name, "w") as f:
        json.dump(save_files, f, indent=4, ensure_ascii=False, separators=(',', ': '))

    print("------",save_files)

    with open(error_file_name, "w") as f:
        json.dump(error_files, f, indent=4, ensure_ascii=False, separators=(',', ': '))





pattern = re.compile(r"\{'name': '([^']+)', 'box': \[([^,]+), ([^,]+), ([^,]+), ([^,]+)\]\}")

parser = argparse.ArgumentParser(description='Args.')
parser.add_argument('--box_dir', type=str, default="/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox_new_generated_6_18/MM_bench/remove", help='root path to the dataset')
parser.add_argument('--save_dir', type=str, default="/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox_new_generated_6_18/bbox_new_clean/MM_bench/remove", help='save_dir')
args = parser.parse_args()

box_dir = args.box_dir
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

for filename in os.listdir(box_dir):
    if filename.endswith('.jsonl'):
        box_json = os.path.join(box_dir, filename)
        per_json_box(box_json, save_dir)

