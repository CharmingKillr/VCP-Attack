import argparse
import os
import random
from PIL import Image
import time

import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
from transformers import AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence


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

    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--img_path", default='your image path', type=str)
    parser.add_argument("--query", default='Briefly describe the content of this image in no more than three sentences.', type=str)

    parser.add_argument("--output_path", default="your output path", type=str)

    parser.add_argument("--txt_name", default='your txt_name.txt', type=str)
    #parser.add_argument("--txt_name", default='FOA_Attack.txt', type=str) # AttackVLM M_Attack AdvDiffVLM Any_attack FOA_Attack

    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    args = parser.parse_args()
    print(args)
    print(f"Loading Ovis2-16b model...")
    os.makedirs(args.output_path, exist_ok=True)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-16B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).to(device)
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
   
    print("Done.")

    # load image
    imagenet_data = ImageFolderWithPaths(args.img_path, transform=None)
    dataloader    = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # img2txt
    for i, (_, image_path) in enumerate(dataloader):
        image_paths = image_path
        start = time.perf_counter()
        print(f"Ovis2-16b img2txt: {i}/{args.num_samples//args.batch_size}")
        if i >= args.num_samples//args.batch_size:
            print(f"Successfully processed {args.num_samples} images to text!")
            break 

        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            query = f'<image>\n{args.query}'
            prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image], max_partition=9)

            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            batch_input_ids.append(input_ids.to(device=model.device))
            batch_attention_mask.append(attention_mask.to(device=model.device))
            batch_pixel_values.append(pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device))

        # pad input_ids & attention_mask
        batch_input_ids = pad_sequence([i.flip([0]) for i in batch_input_ids], batch_first=True, padding_value=0).flip([1])
        batch_input_ids = batch_input_ids[:, -model.config.multimodal_max_length:]
        batch_attention_mask = pad_sequence([i.flip([0]) for i in batch_attention_mask], batch_first=True, padding_value=False).flip([1])
        batch_attention_mask = batch_attention_mask[:, -model.config.multimodal_max_length:]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = model.generate(batch_input_ids,
                                        pixel_values=batch_pixel_values,
                                        attention_mask=batch_attention_mask,
                                        **gen_kwargs)

        # 写入结果
        output_file = os.path.join(args.output_path, args.txt_name)

        with open(output_file, 'a') as f:
            for output in output_ids:
                caption = text_tokenizer.decode(output, skip_special_tokens=True).strip().replace('\n', ' ')
                f.write(caption + '\n')
        
        end = time.perf_counter()
        print(f"query time for {args.batch_size} samples:", (end - start))
        
    print("Caption saved.")