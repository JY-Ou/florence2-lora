import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.utils import GenerationConfig


def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    # base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, tokenizer_class=AutoTokenizer, use_fast=False, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)
    # base.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path,safe_serialization=False)
    # base_tokenizer.save_pretrained(output_path)

    # 删除除了bin文件以外的所有文件
    for file in os.listdir(output_path):
        if not file.endswith('.bin'):
            os.remove(os.path.join(output_path, file))


if __name__ == "__main__":
# OD/lora-test(large)/model_checkpointsb3/epoch_6
    lora_path = "./output/model_checkpoints_lora_all1_o_1/epoch_2"
    model_path = "./model/Florence"
    output = "./model/lora_all1_o_1"

    if not os.path.exists(output):
        os.makedirs(output)
    apply_lora(model_path, output, lora_path)

