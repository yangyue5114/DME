import json
import os

# JSON 文件的路径
# json_file_path = '/mnt/petrelfs/yangyue/SoM/som_remove/new/MMbench/image_gpt_query_info.json'  # 请替换为实际的文件路径
# clean_file_path = '/mnt/petrelfs/yangyue/SoM/som_remove/new/MME/image_clean_info_select_5.json'
json_file_path = '/mnt/petrelfs/yangyue/SoM/som_remove/new/MMbench/image_gpt_query_info.json'  # 请替换为实际的文件路径
clean_file_path = '/mnt/petrelfs/yangyue/SoM/som_remove/new/MMbench/image_clean_info_select_5.json'
success_num = 0
mask_not_exist_num = 0

# 打开文件并读取数据
with open(json_file_path, 'r') as file:
    # 加载 JSON 数据
    all_json_datas = json.load(file)

new_clean_mask_json = []
for per_json_data in all_json_datas:
    per_image_som_path = per_json_data["new_per_image_som_path"]
    mask_dir_name = os.path.dirname(per_json_data["mask_path_list"][0])
    
    with open(clean_file_path, 'r') as file:
        # 加载 JSON 数据
        clean_data = json.load(file)

    # 遍历字典中的每个键
    for clean_data_key, clean_data_value in clean_data.items():
        # 检查键名中是否包含特定字符串 "xxx"
        if per_image_som_path in clean_data_key:
            new_clean_mask_info_list = []
            clean_masks_list = clean_data[per_image_som_path]
            for per_mask_info in clean_masks_list:
                per_mask_path = os.path.join(mask_dir_name, str(per_mask_info["object_mark"])+"-mask.jpg")
                per_mask_name = per_mask_info["object_name"]
                if not os.path.exists(per_mask_path):
                    mask_not_exist_num = mask_not_exist_num + 1
                
                new_clean_mask_one = {"per_mask_path":per_mask_path, "per_mask_name":per_mask_name}
                new_clean_mask_info_list.append(new_clean_mask_one)

            success_num = success_num + 1
            del per_json_data["mask_path_list"]
            per_json_data["mask_info_list"] = new_clean_mask_info_list
    
    new_clean_mask_json.append(per_json_data)

clean_path = json_file_path.replace("image_gpt_query_info","image_to_be_remove")
with open(clean_path, "w") as f:
    json.dump(new_clean_mask_json, f, indent=4, ensure_ascii=False, separators=(',', ': '))

print("success_num:",success_num)
print("mask_not_exist_num:", mask_not_exist_num)
            
                 
            
            