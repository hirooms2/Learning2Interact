import torch
import peft
from utils import load_peft_model, setup_tokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_base_model(model_name):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float16,
        device_map="auto"
    )

    return base_model

def compare_outputs(base_model_name, peft_model_path, prompt="Explain quantum physics in simple terms."):
    # 1. Tokenizer
    tokenizer = setup_tokenizer(base_model_name)

    # 2. Load base model
    base_model = load_base_model(base_model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # 3. Load PEFT model
    peft_model = load_peft_model(base_model, peft_model_path)
    peft_model.eval()

    # 4. Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

    # 5. Generate from base + peft
    with torch.no_grad():
        output_peft = peft_model.generate(**inputs, max_new_tokens=100)
    decoded_peft = tokenizer.decode(output_peft[0], skip_special_tokens=True)

    # 6. Merge and generate from merged model
    merged_model = peft_model.merge_and_unload()
    quant_cfg = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype="bfloat16")
    merged_model.save_pretrained("merged-fp16")
    merged_model = AutoModelForCausalLM.from_pretrained(
        "merged-fp16", quantization_config=quant_cfg, device_map="auto")
    merged_model.save_pretrained("merged-4bit")

    with torch.no_grad():
        output_merged = merged_model.generate(**inputs, max_new_tokens=100)
    decoded_merged = tokenizer.decode(output_merged[0], skip_special_tokens=True)

    # 7. Print and compare
    print("=== base + PEFT output ===")
    print(decoded_peft)

    print("\n=== merged model output ===")
    print(decoded_merged)

    print("\n=== Match: ", decoded_peft == decoded_merged)

if __name__ == "__main__":
    base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"           # or your custom model
    peft_model_path = "/home/user/junpyo/Learning2Interact/model_weights/sft_model_0619104232_sft_train_onyinteraction_dialog-5_epoch5_batch2_gas4_lr3e-5_no_max_length_gpt_cot_turn5_rerank3"        # path to PEFT adapter
    compare_outputs(base_model_name, peft_model_path)
