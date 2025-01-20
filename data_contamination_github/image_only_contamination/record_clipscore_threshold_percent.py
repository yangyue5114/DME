import json
import glob
import  os


def load_json_file(filepath):
    """
    读取指定路径的JSON文件，并将其内容转换为Python字典。
    
    参数:
    - filepath (str): JSON文件的路径。
    
    返回:
    - dict: 从JSON文件中加载的字典。
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def list_json_files(folder_path):
    # 使用 glob 模块列出所有 JSON 文件的完整路径
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    return json_files


def find_query_path_high_threshold(data, min_threshold):
    """
    遍历字典，找出值也是字典且包含'query_path'键的所有键值对。
    
    参数:
    - data (dict): 要遍历的字典。
    
    返回:
    - dict: 一个新字典，只包含满足条件的键值对。
    """
    result = {}
    
    key_num = 0
    total_key_num=0
    total_train_num=0

    for key, value in data.items():
        key_bool=False

        if isinstance(value, dict) and "query_path" in value:
            total_key_num=total_key_num+1
            train_num=0

            query_path = value["query_path"]
            results_data = value["results_data"]
            
            for result_data in results_data:
                if result_data["threshold"] >= min_threshold and int(result_data["num_matches"]) > 0:
                    key_bool=True
                    train_num = train_num + int(result_data["num_matches"])
            total_train_num = total_train_num + train_num
            result[query_path] = {train_num,total_train_num}
            # print(train_num,total_train_num)
        
        if key_bool==True:
            key_num = key_num+1
        

    return result, key_num, total_key_num



def compute_percent(json_path):
    json_data = load_json_file(json_path)
    min_thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]

    for min_threshold in min_thresholds:
        result,key_num, total_key_num= find_query_path_high_threshold(json_data, min_threshold)
        print("min_threshold:",min_threshold,"percent:",key_num/total_key_num*100, "key_num:",key_num,"total_key_num:",total_key_num)

folder_path = '/mnt/petrelfs/yangyue/dynamic_eval/data_contamination/dataset-coco-caption/retrieval/clipscore_results/MME'
json_paths = list_json_files(folder_path)

for json_path in json_paths:
    print(json_path)
    compute_percent(json_path)
# json_path="/mnt/petrelfs/yangyue/dynamic_eval/data_contamination/dataset-cc3m/retrieval/clipscore_results/results_sampled-cc3m_MME_Benchmark_MME_Benchmark_release_version_artwork_images_0507-12-51-42.json"

