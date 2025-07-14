import argparse
import os
import random
import base64
from openai import OpenAI
from tqdm import tqdm
import torch
import torchvision
import numpy as np
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

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def run_gpt4o_captioning(out_file, dataloader=None, prompt=None, max_tokens=None, batch_size=None, num_samples=None):
    client = OpenAI(
        base_url='https://api.openai-proxy.org/v1',
        api_key='your api key',
    )


    for idx, (_, image_path) in enumerate(tqdm(dataloader, desc="Processing images")):
        if idx >= num_samples//batch_size:
            print(f"Successfully processed {args.num_samples} images to text!")
            break 

        image_path = image_path[0]
        image_base64 = encode_image_to_base64(image_path)

        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens
        )
        caption = response.choices[0].message.content.strip().replace('\n', ' ')
        with open(out_file, 'a') as f:
            f.write(caption + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPT-4o Image Captioning Script")
    parser.add_argument("--img_path", default='/data/jw/mnt/mount/VCP-Attack/output_adv_images/AttackVLM', type=str)
    parser.add_argument("--query", default='Briefly describe the content of this image in no more than three sentences.', type=str)

    parser.add_argument("--output_path", default="/data/jw/mnt/mount/VCP-Attack/output_adv_captions_1000/gpt-4o-2024-11-20", type=str)
    #parser.add_argument("--txt_name", default='enhance_pca_10_1000_8_128_eps16_steps240_0.8max.txt', type=str)
    parser.add_argument("--txt_name", default='AttackVLM.txt', type=str) # AttackVLM M_Attack AdvDiffVLM Any_Attack FOA_Attack
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--max_new_tokens", default=128, type=int)

    args = parser.parse_args()
    # load image
    imagenet_data = ImageFolderWithPaths(args.img_path, transform=None)
    dataloader    = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, args.txt_name)

    run_gpt4o_captioning(output_file, dataloader, args.query, args.max_new_tokens, args.batch_size, args.num_samples)
