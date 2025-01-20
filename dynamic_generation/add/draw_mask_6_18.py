import os
from PIL import Image, ImageDraw
import time
import json
import argparse

def make_mask_dir(known_file_path,new_directory):

    # 为每个mask图像生成一个时间戳和标签
    now_T = time.strftime('%m%d-%H-%M-%S', time.localtime())
    # Extract the filename without the extension
    # file_name_without_extension = os.path.splitext(os.path.basename(known_file_path))[0]+'_'+now_T
    file_name_without_extension = os.path.basename(known_file_path).replace('.','_')
    
    # Construct the path for the new subdirectory
    new_subdirectory_path = os.path.join(new_directory, file_name_without_extension)

    # Create the new subdirectory (ensure you have permissions or choose an appropriate location)
    os.makedirs(new_subdirectory_path, exist_ok=True)

    return new_subdirectory_path


def draw_masks(img_file_path,new_sub_mask_dir,objects_details):
    print(img_file_path)
    img = Image.open(img_file_path)

    width, height = img.size
    image_dimensions = (width, height)
    print(image_dimensions)

    mask_path_list=[]

    for index, object_details in enumerate(objects_details):
        # 为每个物体创建一个新的图像
        image = Image.new("RGB", image_dimensions, "black")
        draw = ImageDraw.Draw(image)

        # 根据提供的坐标绘制矩形
        top_left = (object_details['box'][0],object_details['box'][1])
        bottom_right = (object_details['box'][2],object_details['box'][3])
        # print(top_left,bottom_right)
        draw.rectangle([top_left, bottom_right], fill="white")

        
        label = object_details['name']
        file_path = '{}/{}_{}-mask.jpg'.format(new_sub_mask_dir,index,label)
        # print("file_path:",file_path)

        info_dict= {'object_name':label, 'box_mask_path':file_path}
        image.save(file_path)

        mask_path_list.append(info_dict)
    
    return mask_path_list


def run_inpainting_per_image(img_file_path,new_mask_dir,objects_details):
    
    new_sub_mask_dir = make_mask_dir(img_file_path,new_mask_dir)
    # print(new_sub_mask_dir)

    # # 假设objects_details包含所有物体的信息
    # objects_details = [
    # {'name': 'tree', 'box':[170, 7, 315, 59]},
    # {'name': 'bench', 'box':[515, 253, 616, 314]},
    # {'name': 'dog', 'box':[393, 37, 499, 87]},
    # {'name': 'cat', 'box':[39, 99, 155, 187]},
    # {'name': 'bicycle', 'box':[46, 221, 190, 294]},
    # {'name': 'balloon', 'box':[518, 187, 606, 249]},
    # {'name': 'sign', 'box':[95, 341, 166, 426]},
    # {'name': 'flower', 'box':[22, 208, 110, 272]},
    # {'name': 'rock', 'box':[241, 3, 340, 93]},
    # {'name': 'bird', 'box':[97, 42, 222, 136]}
    # ]

    mask_path_list = draw_masks(img_file_path,new_sub_mask_dir,objects_details)

    # print(mask_path_list)

    added_list=[]

    for mask in mask_path_list:
        parameters={'source_img_path':img_file_path, 'object_name':mask['object_name'], 'mask_img_path':mask['box_mask_path']}
        added_list.append(parameters)
    
    # print(added_list)

    parameter_save_path= os.path.join(new_sub_mask_dir,'parameter.json')
    print(parameter_save_path)

    with open(parameter_save_path, "w") as f:
        json.dump(added_list, f, indent=4, ensure_ascii=False, separators=(',', ': '))

    return added_list



def per_task_json(bbox_json_path, save_mask_dir):
    
    task_json_path=bbox_json_path.split('/')[-2]
    save_mask_dir=os.path.join(save_mask_dir,task_json_path)
    os.makedirs(save_mask_dir, exist_ok=True)
    
    print("per_task_json")
    # 打开并读取 JSON 文件
    with open(bbox_json_path, 'r') as file:
        img_datas = json.load(file)


    for img_data in img_datas:
        img_path = img_data['img_path']
        objects_details = img_data['bboxes']
        try:
            # print("try")
            added_list = run_inpainting_per_image(img_path, save_mask_dir, objects_details)
            print("try ok added list")
        except:
            print("except")
            continue




parser = argparse.ArgumentParser(description="Image manipulation command line tool")
parser.add_argument('--clean_bbox_json_dir', type=str, required=True, default='/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox_clean/MME/remove')
parser.add_argument('--save_mask_dir', type=str, required=True, default='/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/ten_masks/MME/remove')

args = parser.parse_args()

# Known file path and the new directory (modify these paths as needed)
# img_file_path = r"mme\existence\000000007816.jpg"
clean_bbox_json_dir = args.clean_bbox_json_dir
save_mask_dir = args.save_mask_dir
os.makedirs(save_mask_dir, exist_ok=True)
# bbox_json_path = '/mnt/petrelfs/yangyue/dynamic_eval/evaluation/generate_mask/bbox_clean/MME/remove/existence_0507-17-35-55/0507-19-58-20_success.json'




for filename in os.listdir(clean_bbox_json_dir):
    # if filename == 'existence':
    #     print("skip!",os.path.join(clean_bbox_json_dir, filename,'success.json'))
    # else:
    clean_json_path = os.path.join(clean_bbox_json_dir, filename,'success.json')
    # print(clean_json_path)

    per_task_json(clean_json_path, save_mask_dir)
# print(added_list)

