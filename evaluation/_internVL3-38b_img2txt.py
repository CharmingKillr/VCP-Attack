import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import random
from PIL import Image
import time

import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
import math 
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T


# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]  # path: str

        # image_processed = vis_processors["eval"](original_tuple[0])
        # text_processed  = txt_processors["eval"](class_text_all[original_tuple[1]])
        
        return original_tuple[1], path
    
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    model_path = "OpenGVLab/InternVL3-38B"
    device_map = {}
    used_devices = [0, 1]  
    world_size = len(used_devices)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_id = used_devices[i]
            device_map[f'language_model.model.layers.{layer_cnt}'] = device_id
            layer_cnt += 1
    vision_device = used_devices[0]
    device_map['vision_model'] = vision_device
    device_map['mlp1'] = vision_device
    device_map['language_model.model.tok_embeddings'] = vision_device
    device_map['language_model.model.embed_tokens'] = vision_device
    device_map['language_model.output'] = vision_device
    device_map['language_model.model.norm'] = vision_device
    device_map['language_model.model.rotary_emb'] = vision_device
    device_map['language_model.lm_head'] = vision_device
    device_map[f'language_model.model.layers.{num_layers - 1}'] = vision_device

    return device_map
    
if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser(description="Demo")
    
    parser.add_argument("--img_path", default='your image path', type=str)
    parser.add_argument("--query", default='Briefly describe the content of this image in no more than three sentences.', type=str)

    parser.add_argument("--output_path", default="your output path", type=str)

    parser.add_argument("--txt_name", default='your txt_name.txt', type=str)
    #parser.add_argument("--txt_name", default='FOA_Attack.txt', type=str) # AttackVLM M_Attack AdvDiffVLM Any_attack FOA_Attack

    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    args = parser.parse_args()
    print(args)
    print(f"Loading InternVL3-38b model...")
    os.makedirs(args.output_path, exist_ok=True)
    
    device_map = split_model('InternVL3-38B')

    path = "OpenGVLab/InternVL3-38B"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map=device_map).eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
   
    print("Done.")

    # load image
    imagenet_data = ImageFolderWithPaths(args.img_path, transform=None)
    dataloader    = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # img2txt
    for i, (_, image_path) in enumerate(dataloader):
        image_paths = image_path
        start = time.perf_counter()
        
        print(f"InternVL3-38b img2txt: {i}/{args.num_samples//args.batch_size}")
        if i >= args.num_samples//args.batch_size:
            print(f"Successfully processed {args.num_samples} images to text!")
            break 
        # 加载所有图像并拼接
        pixel_values_list = []
        num_patches_list = []

        for image_path in image_paths:
            pixel_values_i = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values_i)
            num_patches_list.append(pixel_values_i.size(0))

        # 拼接所有图片patch
        pixel_values = torch.cat(pixel_values_list, dim=0)
        
        # 构造 query 列表
        questions = [args.query] * len(num_patches_list)
        generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=True)

        # 批量对话生成
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config
        )
        # 写入结果
        processed_captions = [r.strip().replace('\n', ' ') for r in responses]
        output_file = os.path.join(args.output_path, args.txt_name)

        with open(output_file, 'a') as f:
            for caption in processed_captions:
                f.write(caption + '\n')
        
        end = time.perf_counter()
        print(f"query time for {args.batch_size} samples:", (end - start))
        
    print("Caption saved.")