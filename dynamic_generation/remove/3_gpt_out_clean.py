import json
import re



# 假设 json 文件路径为 'data.json'
file_path = '/mnt/petrelfs/yangyue/SoM/som_remove/new/MMbench/image_gpt_query_info_select_5.json'
info_dict ={}

success_num = 0
error_num = 0

# 打开文件
with open(file_path, 'r') as file:
    # 遍历文件中的每一行
    for line in file:
        # 去除每行的首尾空白字符
        line = line.strip()

        # 打印这一行，检查其格式
        # print("原始行内容:", line)

        # 尝试解析这一行的 JSON 数据
        try:
            data_dict = json.loads(line)
            # print("解析后的字典:", data_dict, type(data_dict))

            data_dict = json.loads(data_dict)
            # print("解析后的字典:", data_dict, type(data_dict))
            # 提取 gpt_response 字段的内容
            gpt_response = data_dict["gpt_response"]
            per_image_som_path = data_dict["per_image_som_path"]
            fixed_str_data = re.sub(r'(?<=\: )(\b\w+\b)(?=})', r'"\1"', gpt_response)
            gpt_response_json = re.sub(r'(?<=\: )(\b\w+\b)(?=,)', r'"\1"', fixed_str_data)
            
            gpt_response_json = json.loads(gpt_response_json)
            print(type(gpt_response_json),type(per_image_som_path))
            
            info_dict [per_image_som_path] = gpt_response_json
            success_num = success_num +1
            # print(info_dict["/mnt/petrelfs/yangyue/SoM/som_remove/SEEDBench/Visual_Reasoning/16719_jpg/som_result.jpg"][0]["object_mark"])

            # 打印转换后的 JSON 对象
            # print(json.dumps(gpt_response, indent=4))
            
        except Exception as e:
            error_num = error_num + 1
            print(f"未知错误: {e}")



clean_path = file_path.replace("image_gpt_query_info_select_5","image_clean_info_select_5")
with open(clean_path, "w") as f:
    json.dump(info_dict, f, indent=4, ensure_ascii=False, separators=(',', ': '))


print("success_num", success_num)
print("error_num", error_num)





# # import json

# # 输入的原始字符串
# data = "{\"per_image_som_path\": \"/mnt/petrelfs/yangyue/SoM/som_remove/SEEDBench/Visual_Reasoning/16719_jpg/som_result.jpg\", \"per_question\": \"What advice would you give to the couple if they want to enhance their dance performance in this kind of environment?\", \"gpt_response\": \"[\\n    {\\\"object_mark\\\": 22, \\\"object_name\\\": \\\"object\\\"},\\n    {\\\"object_mark\\\": 23, \\\"object_name\\\": \\\"object\\\"},\\n    {\\\"object_mark\\\": 33, \\\"object_name\\\": \\\"object\\\"},\\n    {\\\"object_mark\\\": 32, \\\"object_name\\\": \\\"object\\\"},\\n    {\\\"object_mark\\\": 39, \\\"object_name\\\": \\\"object\\\"}\\n]\"}"

# # 将字符串转换为字典
# data_dict = json.loads(data)

# # 提取 gpt_response 字段的内容
# gpt_response = data_dict["gpt_response"]

# # 将 gpt_response 的字符串转换为 JSON 对象
# gpt_response_json = json.loads(gpt_response)

# # 打印转换后的 JSON 对象
# print(json.dumps(gpt_response_json, indent=4))

