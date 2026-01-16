import os
import gc
import torch
import io
from huggingface_hub import create_repo, HfApi
from utils import setup_multimodal_model, load_hf_model_and_processor_or_tokenizer

MULTIMODAL_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
PRETRAINED_MODEL_NAME = "lmsys/vicuna-7b-v1.5"
PRETRAINED_PROJECTOR_NAME_ORIGINAL = (
    "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"
)
PRETRAINED_PROJECTOR_NAME_LORENZO = (
    "lbasile/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"
)
CACHE_DIR = os.path.expanduser("~/scratch/huggingface/hub/")


def push_models_to_hub():
    print("Setting up original pretrained model...")
    model, processor = setup_multimodal_model(
        multimodal_model_name_or_path=MULTIMODAL_MODEL_NAME,
        model_cache_dir=CACHE_DIR,
        language_model_name_or_path=PRETRAINED_MODEL_NAME,
        pretrained_projector_name_or_path=PRETRAINED_PROJECTOR_NAME_ORIGINAL,
        replace_language_model=True,
        replace_pretrained_projector=True,
        device_map="cpu",
        attn_implementation="sdpa",
        model_dtype=torch.float16,
        skip_processor=False,
    )

    print("Uploading original pretrained model to Hugging Face Hub...")
    new_repo_id = "l11p/llava-v1.5-7b-pretrained-liuhaotian"
    create_repo(new_repo_id, exist_ok=True)

    readme_content = """---
tags:
- image-text-to-text
- llava
- multimodal
- vision-language
model-index:
- name: llava-v1.5-7b-pretrained-liuhaotian
  results: []
---

# LLaVA v1.5 7B Pretrained Original

Original pretrained LLaVA v1.5 7B model with Vicuna 1.5 LM and pretrained projector.
"""
    model.push_to_hub(new_repo_id, model_card=readme_content)
    processor.push_to_hub(new_repo_id)

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    print("Setting up Lorenzo's pretrained model...")
    model, processor = setup_multimodal_model(
        multimodal_model_name_or_path=MULTIMODAL_MODEL_NAME,
        model_cache_dir=CACHE_DIR,
        language_model_name_or_path=PRETRAINED_MODEL_NAME,
        pretrained_projector_name_or_path=PRETRAINED_PROJECTOR_NAME_LORENZO,
        replace_language_model=True,
        replace_pretrained_projector=True,
        device_map="cpu",
        attn_implementation="sdpa",
        model_dtype=torch.float16,
        skip_processor=False,
    )

    print("Uploading Lorenzo's pretrained model to Hugging Face Hub...")
    new_repo_id = "l11p/llava-v1.5-7b-pretrained-lbasile"
    create_repo(new_repo_id, exist_ok=True)
    model.push_to_hub(new_repo_id)
    processor.push_to_hub(new_repo_id)

    readme_content = """---
tags:
- image-text-to-text
- llava
- multimodal
- vision-language
model-index:
- name: llava-v1.5-7b-pretrained-lbasile
  results: []
---

# LLaVA v1.5 7B Pretrained Lorenzo Basile

Lorenzo Basile's (lbasile) pretrained LLaVA v1.5 7B model with Vicuna 1.5 LM and pretrained projector.
"""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=new_repo_id,
        repo_type="model",
    )

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()


def main():
    push_models_to_hub()

if __name__ == "__main__":
    main()
