import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
# from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
from datetime import datetime
from petrel_client.client import Client
import json
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the CLIP model
model_ID = "/mnt/petrelfs/yangyue/dynamic_eval/data_contamination/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_ID).to(device)
bscnt = 0
preprocess = CLIPImageProcessor.from_pretrained(model_ID)

def ceph_clip_extrct_feature(image_input):
    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embeddings = model.get_image_features(image_input)

    return embeddings

def clip_img_score(embedding_a,embedding_b):
    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    return similarity_score.item()


def ssim_img_score(img1_path,img2_path):
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))
    ssim_score = ssim(img1, img2, channel_axis=2,data_range=255)
    return ssim_score

def prepare_image_tensor(image_array):
    # """将numpy图像数组转换为PyTorch张量，并调整维度顺序。"""
    preprocess_img = []
    for image in image_array:
        tmp_img = preprocess(image, return_tensors="pt")["pixel_values"]
        preprocess_img.append(tmp_img)
    image_tensor = torch.cat(preprocess_img).to(device)
    return image_tensor



def save_progress(index, progress_file_path):
    # """
    # 保存处理进度到指定的文件中。

    # :param index: 当前处理到的索引。
    # :param progress_file_path: 进度文件的路径。
    # """
    with open(progress_file_path, "w") as f:
        json.dump({"save_index": index}, f)

def load_progress(progress_file_path):
    # """
    # 从指定的文件加载处理进度。

    # :param progress_file_path: 进度文件的路径。
    # :return: 加载的进度(索引)如果文件不存在,则返回0。
    # """
    if os.path.exists(progress_file_path):
        with open(progress_file_path, "r") as f:
            progress = json.load(f)
            return progress["save_index"]
    return 0


def append_error_log(img_url, error_log_path):
    """
    将错误的img_url追加到指定的错误日志文件中。
    
    :param img_url: 出错的图片URL
    :param error_log_path: 错误日志文件的路径
    """
    with open(error_log_path, 'a', encoding='utf-8') as f:
        # 使用json.dumps将img_url转换为JSON格式的字符串，然后写入文件
        # 每条记录占一行
        f.write(json.dumps(img_url, ensure_ascii=False) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default=None, help='data root path')
    args = parser.parse_args()

    files = json.load(open(args.json_path))
    
    
    conf_path = '~/petreloss.conf'
    client = Client(conf_path) # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件

    #files = client.get_file_iterator("s3://evaluation_data/evaluation_datasest_liushuo/classification/classification/BIRDS")
    #files = client.get_file_iterator("s3://public-dataset/laion-coco/images")

    # 定义每批处理的图片数量
    batch_size = 256
    save_num = 25600
    batch_index = 0  # 批次计数

    # 初始化存储所有嵌入的列表和URL到索引的字典
    embeddings_list = []  # 使用列表存储所有嵌入
    url_to_index = {}  # 字典映射 img_url 到其在列表中的索引

    
    save_emb_root = './test_embeddings_spot'
    save_json_root = './test_json_files_spot'
    os.makedirs(save_emb_root, exist_ok=True)
    os.makedirs(save_json_root, exist_ok=True)
    
    sub_dir = args.json_path.split('/')[-1].split('.')[0]

    save_emb_path = os.path.join(save_emb_root, sub_dir)
    save_json_path = os.path.join(save_json_root, sub_dir)
    
    os.makedirs(save_emb_path, exist_ok=True)
    os.makedirs(save_json_path, exist_ok=True)
    # 确保保存的文件夹存在

    error_log_path = os.path.join(save_json_path, "1_error_list.json")
    progress_file_path = os.path.join(save_json_path, "1_spot_progress.json")

    

    save_index = load_progress(progress_file_path)
    start_index = save_index*save_num

    save_img = 0
    cnt = 0
    current_img_list = []
    current_img_url = []
    save_img_file = []
    save_dict = {}
    # error_list = []
    # 开始处理文件
    for index, img_url in enumerate(files[start_index:], start=start_index):
        try:
            img_bytes = client.get(img_url)
            assert(img_bytes is not None)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb_transposed = img_rgb.transpose(2, 0, 1)
        except Exception as e:
            # error_list.append(img_url)
            append_error_log(img_url, error_log_path)
            continue
        current_img_list.append(img_rgb_transposed)
        current_img_url.append(img_url)
        save_img_file.append(img_url)
        cnt+=1
        save_img+=1
        if save_img >= save_num:
            img_tensor = prepare_image_tensor(current_img_list)
            embeddings = ceph_clip_extrct_feature(img_tensor)
            for url, emb in zip(current_img_url, embeddings):
                save_dict[url] = emb.unsqueeze(0)
            torch.save(save_dict, os.path.join(save_emb_path, f'embeddings_tensor_{save_index}.pt'))
            with open(os.path.join(save_json_path, f"files_{save_index}.json"), 'w') as f:
                json.dump(save_img_file, f, ensure_ascii=False, separators=(',',';'), indent=4)
            save_img = 0
            save_index += 1
            cnt, current_img_list, current_img_url = 0, [], []
            save_img_file, save_dict, save_img = [], {}, 0

            print("saved:", index)
            print("save_index:", save_index)
            save_progress(save_index,progress_file_path)

        elif cnt >= batch_size:
            img_tensor = prepare_image_tensor(current_img_list)
            embeddings = ceph_clip_extrct_feature(img_tensor)
            for url, emb in zip(current_img_url, embeddings):
                save_dict[url] = emb.unsqueeze(0)
            cnt, current_img_list, current_img_url = 0, [], []

            print("index:", index)
            print("save_index:", save_index)
            save_progress(save_index,progress_file_path)

    img_tensor = prepare_image_tensor(current_img_list)
    embeddings = ceph_clip_extrct_feature(img_tensor)
    for url, emb in zip(current_img_url, embeddings):
        save_dict[url] = emb
    torch.save(save_dict, os.path.join(save_emb_path, f'embeddings_tensor_{save_index}.pt'))
    with open(os.path.join(save_json_path, f"files_{save_index}.json"), 'w') as f:
        json.dump(save_img_file, f, ensure_ascii=False, separators=(',',';'), indent=4)

    # with open(os.path.join(save_json_path, "error_list.json"), 'w') as f:
    #     json.dump(error_list, f, ensure_ascii=False, separators=(',',';'), indent=4)
