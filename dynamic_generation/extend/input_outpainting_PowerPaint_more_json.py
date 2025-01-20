import random
import argparse
import cv2
import gradio as gr
import numpy as np
import torch
from controlnet_aux import HEDdetector, OpenposeDetector
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from PIL import Image, ImageFilter
from pipeline.pipeline_PowerPaint import \
    StableDiffusionInpaintPipeline as Pipeline
from pipeline.pipeline_PowerPaint_ControlNet import \
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from utils.utils import TokenizerWrapper, add_tokens
import time
import json
import os

torch.set_grad_enabled(False)

# 模型初始化: 脚本初始化了一个用于图像修复的管道 StableDiffusionInpaintPipeline。
weight_dtype = torch.float16
global pipe
pipe = Pipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting',
    torch_dtype=weight_dtype)
pipe.tokenizer = TokenizerWrapper(
    from_pretrained='runwayml/stable-diffusion-v1-5',
    subfolder='tokenizer',
    revision=None)
print("------model loaded------")
add_tokens(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    placeholder_tokens=['P_ctxt', 'P_shape', 'P_obj'],
    initialize_tokens=['a', 'a', 'a'],
    num_vectors_per_token=10)

# 加载模型: 使用 load_model 从保存的张量文件中加载特定的模型组件（unet 和 text_encoder）。
from safetensors.torch import load_model
load_model(pipe.unet, "/mnt/petrelfs/yangyue/dynamic_eval/evaluation/inpainting/PowerPaint/models/unet/unet.safetensors")
load_model(pipe.text_encoder, "/mnt/petrelfs/yangyue/dynamic_eval/evaluation/inpainting/PowerPaint/models/text_encoder/text_encoder.safetensors")
pipe = pipe.to('cuda')


# 额外模型: 加载了深度估计和特征提取模型（DPTForDepthEstimation 和 DPTFeatureExtractor），以及自定义检测器（HEDdetector 和 OpenposeDetector）。
depth_estimator = DPTForDepthEstimation.from_pretrained(
    'Intel/dpt-hybrid-midas').to('cuda')
feature_extractor = DPTFeatureExtractor.from_pretrained(
    'Intel/dpt-hybrid-midas')
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

global current_control
current_control = 'canny'
# controlnet_conditioning_scale = 0.8


# 实用函数：设置种子与深度图计算: 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_depth_map(image):
    image = feature_extractor(
        images=image, return_tensors='pt').pixel_values.to('cuda')
    with torch.no_grad(), torch.autocast('cuda'):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode='bicubic',
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


# 图像操纵函数：add prompt
def add_task(prompt, negative_prompt, control_type):
    # print(control_type)
    if control_type == 'object-removal':
        promptA = 'empty scene blur ' + prompt + ' P_ctxt'
        promptB = 'empty scene blur ' + prompt + ' P_ctxt'
        negative_promptA = negative_prompt + ' P_obj'
        negative_promptB = negative_prompt + ' P_obj'
    elif control_type == 'shape-guided':
        promptA = prompt + ' P_shape'
        promptB = prompt + ' P_ctxt'
        negative_promptA = negative_prompt + ', worst quality, low quality, normal quality, bad quality, blurry P_shape'
        negative_promptB = negative_prompt + ', worst quality, low quality, normal quality, bad quality, blurry P_ctxt'
    elif control_type == 'image-outpainting':
        promptA = 'empty scene ' + prompt + ' P_ctxt'
        promptB = 'empty scene ' + prompt + ' P_ctxt'
        negative_promptA = negative_prompt + ' P_obj'
        negative_promptB = negative_prompt + ' P_obj'
    else:
        promptA = prompt + ' P_obj'
        promptB = prompt + ' P_obj'
        negative_promptA = negative_prompt + ', worst quality, low quality, normal quality, bad quality, blurry, P_obj'
        negative_promptB = negative_prompt + ', worst quality, low quality, normal quality, bad quality, blurry, P_obj'

    return promptA, promptB, negative_promptA, negative_promptB


# 图像操纵函数：outpainting
def predict(input_image, prompt, fitting_degree, ddim_steps, scale, seed,
            negative_prompt, task,vertical_expansion_ratio,horizontal_expansion_ratio):
    size1, size2 = input_image['image'].convert('RGB').size

    # 调整图像大小，640/512
    if task =='image-outpainting' and (size1>640 or size2>640):
        if size1 < size2:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (640, int(size2 / size1 * 640)))
        else:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (int(size1 / size2 * 640), 640))
    # elif (size1>2000 or size2>2000):
    #     if size1 < size2:
    #         input_image['image'] = input_image['image'].convert('RGB').resize(
    #             (512, int(size2 / size1 * 512)))
    #     else:
    #         input_image['image'] = input_image['image'].convert('RGB').resize(
    #             (int(size1 / size2 * 512), 512))
    
    # outpainting加上要拓展的地方的图跟mask
    if vertical_expansion_ratio!=None and horizontal_expansion_ratio!=None:
        o_W,o_H = input_image['image'].convert('RGB').size
        c_W = int(horizontal_expansion_ratio*o_W)
        c_H = int(vertical_expansion_ratio*o_H)

        expand_img = np.ones((c_H, c_W,3), dtype=np.uint8)*127
        original_img = np.array(input_image['image'])
        expand_img[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = original_img

        blurry_gap = 10

        expand_mask = np.ones((c_H, c_W,3), dtype=np.uint8)*255
        if vertical_expansion_ratio == 1 and horizontal_expansion_ratio!=1:
            expand_mask[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio!=1:
            expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio==1:
            expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = 0
        
        input_image['image'] = Image.fromarray(expand_img)
        input_image['mask'] = Image.fromarray(expand_mask)

        

    # 得到整理后的prompt跟negative prompt，并输出
    promptA, promptB, negative_promptA, negative_promptB = add_task(
        prompt, negative_prompt, task)
    print(promptA, promptB, negative_promptA, negative_promptB)
    img = np.array(input_image['image'].convert('RGB'))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image['image'] = input_image['image'].resize((H, W))
    input_image['mask'] = input_image['mask'].resize((H, W))
    set_seed(seed)
    global pipe
    result = pipe(
        promptA=promptA,
        promptB=promptB,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image['image'].convert('RGB'),
        mask_image=input_image['mask'].convert('RGB'),
        width=H,
        height=W,
        guidance_scale=scale,
        num_inference_steps=ddim_steps).images[0]
    mask_np = np.array(input_image['mask'].convert('RGB'))
    red = np.array(result).astype('float') * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
         mask_np.astype('float') / 512.0 * red).astype('uint8'))
    m_img = input_image['mask'].convert('RGB').filter(
        ImageFilter.GaussianBlur(radius=3))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))

    # 用户定义的掩码区域 + 处理后图像与掩码融合的结果
    dict_res = [input_image['mask'].convert('RGB'), result_m]

    # 处理前图像 + 处理前后图像的融合
    dict_out = [input_image['image'].convert('RGB'), result_paste]

    return dict_out, dict_res


# 图像操纵函数：ControlNet 集成
def predict_controlnet(input_image, input_control_image, control_type, prompt,
                       ddim_steps, scale, seed, negative_prompt,controlnet_conditioning_scale):
    promptA = prompt + ' P_obj'
    promptB = prompt + ' P_obj'
    negative_promptA = negative_prompt 
    negative_promptB = negative_prompt 
    size1, size2 = input_image['image'].convert('RGB').size

    if size1 < size2:
        input_image['image'] = input_image['image'].convert('RGB').resize(
            (640, int(size2 / size1 * 640)))
    else:
        input_image['image'] = input_image['image'].convert('RGB').resize(
            (int(size1 / size2 * 640), 640))
    img = np.array(input_image['image'].convert('RGB'))
    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image['image'] = input_image['image'].resize((H, W))
    input_image['mask'] = input_image['mask'].resize((H, W))

    global current_control
    global pipe

    base_control = ControlNetModel.from_pretrained(
        'lllyasviel/sd-controlnet-canny', torch_dtype=weight_dtype)
    control_pipe = controlnetPipeline(pipe.vae, pipe.text_encoder,
                                      pipe.tokenizer, pipe.unet, base_control,
                                      pipe.scheduler, None, None, False)
    control_pipe = control_pipe.to('cuda')
    current_control = 'canny'
    if current_control != control_type:
        if control_type == 'canny' or control_type is None:
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-canny', torch_dtype=weight_dtype)
        elif control_type == 'pose':
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-openpose', torch_dtype=weight_dtype)
        elif control_type == 'depth':
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-depth', torch_dtype=weight_dtype)
        else:
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-hed', torch_dtype=weight_dtype)
        control_pipe = control_pipe.to('cuda')
        current_control = control_type

    controlnet_image = input_control_image
    if current_control == 'canny':
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = np.array(controlnet_image)
        controlnet_image = cv2.Canny(controlnet_image, 100, 200)
        controlnet_image = controlnet_image[:, :, None]
        controlnet_image = np.concatenate(
            [controlnet_image, controlnet_image, controlnet_image], axis=2)
        controlnet_image = Image.fromarray(controlnet_image)
    elif current_control == 'pose':
        controlnet_image = openpose(controlnet_image)
    elif current_control == 'depth':
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = get_depth_map(controlnet_image)
    else:
        controlnet_image = hed(controlnet_image)

    mask_np = np.array(input_image['mask'].convert('RGB'))
    controlnet_image = controlnet_image.resize((H, W))
    set_seed(seed)
    result = control_pipe(
        promptA=promptB,
        promptB=promptA,
        tradoff=1.0,
        tradoff_nag=1.0,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image['image'].convert('RGB'),
        mask_image=input_image['mask'].convert('RGB'),
        control_image=controlnet_image,
        width=H,
        height=W,
        guidance_scale=scale,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        num_inference_steps=ddim_steps).images[0]
    red = np.array(result).astype('float') * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
         mask_np.astype('float') / 512.0 * red).astype('uint8'))

    mask_np = np.array(input_image['mask'].convert('RGB'))
    m_img = input_image['mask'].convert('RGB').filter(
        ImageFilter.GaussianBlur(radius=4))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))
    return [input_image['image'].convert('RGB'), result_paste], [controlnet_image, result_m]


def infer(input_image, text_guided_prompt, text_guided_negative_prompt,
          shape_guided_prompt, shape_guided_negative_prompt, fitting_degree,
          ddim_steps, scale, seed, task, enable_control, input_control_image,
          control_type,vertical_expansion_ratio,horizontal_expansion_ratio,outpaint_prompt,outpaint_negative_prompt,controlnet_conditioning_scale,removal_prompt,removal_negative_prompt):
    
    # print("---输入格式---",type(input_image),input_image.keys())
    # print("---image输入格式---",type(input_image['image']),input_image['image'])
    # print("---image输入格式---",type(input_image['mask']),input_image['mask'])

    if task == 'text-guided':
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt
    elif task == 'shape-guided':
        prompt = shape_guided_prompt
        negative_prompt = shape_guided_negative_prompt
    elif task == 'object-removal':
        prompt = removal_prompt
        negative_prompt = removal_negative_prompt
    elif task == 'image-outpainting':
        prompt = outpaint_prompt
        negative_prompt = outpaint_negative_prompt
        return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
                       seed, negative_prompt, task,vertical_expansion_ratio,horizontal_expansion_ratio)
        # image-outpainting
    else:
        task = 'text-guided'
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt

    if enable_control and task == 'text-guided':
        return predict_controlnet(input_image, input_control_image,
                                  control_type, prompt, ddim_steps, scale,
                                  seed, negative_prompt,controlnet_conditioning_scale)
    else:
        return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
                       seed, negative_prompt, task,None,None)
        # object-removal or text-guided or shape-guided


def select_tab_text_guided():
    return 'text-guided'


def select_tab_object_removal():
    return 'object-removal'

def select_tab_image_outpainting():
    return 'image-outpainting'


def select_tab_shape_guided():
    return 'shape-guided'


def make_result_dir_per_img(known_file_path,new_directory):

    # 为每个mask图像生成一个时间戳和标签
    # now_T = time.strftime('%m%d-%H-%M-%S', time.localtime())
    # Extract the filename without the extension
    file_name_without_extension = os.path.splitext(os.path.basename(known_file_path))[0]
    
    # Construct the path for the new subdirectory
    per_img_result_dir = os.path.join(new_directory, file_name_without_extension)

    # Create the new subdirectory (ensure you have permissions or choose an appropriate location)
    os.makedirs(per_img_result_dir, exist_ok=True)

    return per_img_result_dir

def find_parameter_json_files(main_directory):
    # 创建一个空列表以存储找到的文件路径
    parameter_json_files_paths = []

    # 遍历总目录中的所有项
    for item in os.listdir(main_directory):
        # 构造每一项的完整路径
        item_path = os.path.join(main_directory, item)
        # 检查此路径是否为目录
        if os.path.isdir(item_path):
            # 构造parameter.json的路径
            json_file_path = os.path.join(item_path, 'parameter.json')
            # 检查parameter.json文件是否存在
            if os.path.isfile(json_file_path):
                # 如果文件存在，添加到列表中
                parameter_json_files_paths.append(json_file_path)

    return parameter_json_files_paths










parser = argparse.ArgumentParser(description="Image manipulation command line tool")
# parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
parser.add_argument('--text_guided_prompt', type=str, default="", help='Prompt to guide.')
parser.add_argument('--task', type=str, required=False, default='image-outpainting', help="['text-guided', 'object-removal', 'shape-guided', 'image-outpainting']")
parser.add_argument('--vertical_expansion_ratio', type=float, default=1.5, required=True, help='Prompt to guide.')
parser.add_argument('--horizontal_expansion_ratio', type=float, default=1.5, required=True, help='Prompt to guide.')
parser.add_argument('--result_save_dir', type=str, required=True, help='result_save_dir')
parser.add_argument('--dataset_task_img_json', type=str, required=True, help='task_img_json')
args = parser.parse_args()


# mme_task_img_json = '/mnt/petrelfs/yangyue/dynamic_eval/evaluation/inpainting/PowerPaint/dataset_json_files/pairs_information_3.json'
mme_task_img_json = args.dataset_task_img_json
# result_save_dir = '/mnt/petrelfs/yangyue/dynamic_eval/evaluation/inpainting/PowerPaint/outpainting_task_results/MME/1_5'
result_save_dir = args.result_save_dir
os.makedirs(result_save_dir, exist_ok=True)
# parameter_json_files_paths = find_parameter_json_files(main_mme_task_dir)
# print(parameter_json_files_paths)


with open(mme_task_img_json, 'r') as file:
    original_data_tasks = json.load(file)

data_tasks = original_data_tasks

for task_name, task_img_path_datas in data_tasks.items():
    print(task_name)

    # if task_name!='existence':

    clip_infos = []
    per_img_result_dir = ""

    for img_path_data in task_img_path_datas:
        # image_path = img_path_data['img_path']

        if 'img_path' in img_path_data:
            image_path = img_path_data['img_path']
        elif 'index_path' in img_path_data:
            image_path = img_path_data['index_path']
        elif 'image' in img_path_data:
            image_path = img_path_data['image'].replace("s3://coco_caption/val2014","/mnt/petrelfs/yangyue/prepare_rbt_dynamic_eval/exp_1_more_vlm_task/image_caption/coco_caption_100")
        else:
            assert 1==2


        clip_source_img = image_path  #计算clipscore时记录的json
        text_guided_prompt = args.text_guided_prompt
        mask_path = '/mnt/petrelfs/yangyue/dynamic_eval/evaluation/inpainting/BrushNet/examples/brushnet/src/mask_imgs/dog_0410-22-23-35_mask.jpg'
        mask_image = Image.open(mask_path)
        image_image = Image.open(image_path)
        input_image = {'image':image_image, 'mask':mask_image}

        text_guided_negative_prompt=""    #初始化为空字符串
        shape_guided_prompt=""
        shape_guided_negative_prompt=""
        fitting_degree = 1                #给一个初始化的值
        ddim_steps = 45               #给一个初始化的值
        scale= 7.5                     #给一个初始化的值
        seed = random.randint(0, 2147483647)        #随机一个初始化的值
        enable_control = False                    #不用controlnet，应该用不上
        input_control_image = None             #不用controlnet，应该用不上
        control_type = 'canny'        #给一个初始化的值，应该用不上
        outpaint_prompt = ""
        outpaint_negative_prompt = ""
        controlnet_conditioning_scale = 0.5
        removal_prompt = ""
        removal_negative_prompt = ""

        
        inpaint_result, gallery= infer(input_image, text_guided_prompt, text_guided_negative_prompt,
                shape_guided_prompt, shape_guided_negative_prompt, fitting_degree,
                ddim_steps, scale, seed, args.task, enable_control, input_control_image,
                control_type,args.vertical_expansion_ratio,args.horizontal_expansion_ratio,outpaint_prompt,outpaint_negative_prompt,controlnet_conditioning_scale,removal_prompt,removal_negative_prompt)

        # print(type(inpaint_result),inpaint_result)
        # print(type(gallery),gallery)

        per_task_result_save_dir = os.path.join(result_save_dir,task_name)
        os.makedirs(per_task_result_save_dir, exist_ok=True)

        per_img_result_dir = make_result_dir_per_img(image_path,per_task_result_save_dir)

        for index, item in enumerate(inpaint_result):
                save_inpaint_jpg = f'{per_img_result_dir}/inpaint_result_{index}.jpg'
                item.save(save_inpaint_jpg, 'JPEG')
                if index==1:
                    clip_pair={'clip_source_img':clip_source_img,'clip_result_img':save_inpaint_jpg}
                    clip_infos.append(clip_pair)
                    img_path_data['outpainted_img_path']=save_inpaint_jpg

        for index, item in enumerate(gallery):
            save_gallery_jpg = f'{per_img_result_dir}/gallery_result_{index}.jpg'
            item.save(save_gallery_jpg, 'JPEG')

        


    with open(f"{per_task_result_save_dir}/clip_compute_info.json", "w") as f:
            json.dump(clip_infos, f, indent=4, ensure_ascii=False, separators=(',', ': '))


save_dataset_name = os.path.basename(mme_task_img_json).split('.')[0]
if "_sample10_for_box" in save_dataset_name:
    save_dataset_name = save_dataset_name.replace("_sample10_for_box", "")

with open(f"{result_save_dir}/{save_dataset_name}_outpainted_pairs_information.json", "w") as f:
    json.dump(data_tasks, f, indent=4, ensure_ascii=False, separators=(',', ': '))
