import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
import random
from PIL import Image
import time

import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# from transformers.modeling_utils import infer_auto_device_map
# from accelerate import init_empty_weights

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
    print(f"Loading Qwen25-72b model...")
    os.makedirs(args.output_path, exist_ok=True)
    
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
    )

    min_pixels = 256*28*28
    max_pixels = 299*299

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
   
    print("Done.")

    # load image
    imagenet_data = ImageFolderWithPaths(args.img_path, transform=None)
    dataloader    = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # img2txt
    for i, (_, image_path) in enumerate(dataloader):
        image_paths = image_path  # only one image per batch
        start = time.perf_counter()
        
        print(f"Qwen25-72b img2txt: {i}/{args.num_samples//args.batch_size}")
        if i >= args.num_samples//args.batch_size:
            print(f"Successfully processed {args.num_samples} images to text!")
            break 
        
        messages_batch = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": args.query}
                ]
            }]
            for path in image_paths
        ]

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]

        image_inputs, video_inputs = process_vision_info(messages_batch)

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # write captions
        processed_captions = [c.strip().replace('\n', ' ') for c in output_text]

        output_file = os.path.join(args.output_path, args.txt_name)

        with open(output_file, 'a') as f:
            f.write('\n'.join(processed_captions) + '\n')
        
        end = time.perf_counter()
        print(f"query time for {args.batch_size} samples:", (end - start))
        
    print("Caption saved.")