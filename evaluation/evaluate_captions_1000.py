from modelscope import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import pandas as pd

model_name = "Qwen/Qwen2.5-14B-Instruct"
device_id = 3
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="float16",
    device_map=f"cuda:{device_id}"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--attack_text_path", default="your text path", type=str)

    parser.add_argument("--map_csv_path", default="self_target_classes.csv", type=str)
    parser.add_argument("--output_path", default="your output path", type=str)

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_path ,args.attack_text_path.split('/')[-2] + '_' + args.attack_text_path.split('/')[-1])
    print(args)

    with open(args.attack_text_path, 'r', encoding='utf-8') as f:
        attack_caps = [l.strip() for l in f]
    count_num = len(attack_caps)
    mapping_df = pd.read_csv(args.map_csv_path, encoding='utf-8')

    for i, attack_c in enumerate(attack_caps):
        self_class   = mapping_df.loc[i, 'Self class']
        target_class = mapping_df.loc[i, 'Target class']
        
        prompt_base = f'''
        You are an adversarial caption detector.

        Given the following image description:
        "{attack_c}"

        Determine which class the caption semantically leans toward, based on:
        - Mentioned objects or scenes
        - Synonyms or related terms
        - Overall topic or context

        Options:
        A) {self_class}  
        B) {target_class}  

        Only reply with "A" or "B".
        '''
        
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt_base}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Generated caption {i+1}: {response}")
        with open(args.output_path, 'a') as f:
            response = response.replace('\n', ' ').replace('Output:', '').replace('\"', '').strip()
            f.write(response + '\n')
        
    with open(args.output_path, 'r') as f:
        count = sum(1 for line in f if line.strip() == "B")
        print(f"ASR : {count/count_num}")
        f.close()
    with open(args.output_path, 'a') as f:
        f.write(f"ASR : {count/count_num}")
        f.close()