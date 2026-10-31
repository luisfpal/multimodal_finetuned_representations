"""Build the "pretrained" LLaVA-1.5 reference checkpoint (before visual instruction tuning) and push it to the Hub.

Starts from llava-hf/llava-1.5-{N}b-hf, swaps in the pretrain-stage projector and the base
lmsys/vicuna-{N}b-v1.5 language model, then uploads model and processor to the target repo.
"""

import gc
import io
import os
import torch

from huggingface_hub import HfApi, create_repo

from utils import load_hf_model_and_processor_or_tokenizer
from utils.model_utils import replace_multimodal_projector, replace_multimodal_lm


NUM_PARAMETERS = "13"
BASE_MODEL_NAME = f"llava-hf/llava-1.5-{NUM_PARAMETERS}b-hf"
LANGUAGE_MODEL_NAME = f"lmsys/vicuna-{NUM_PARAMETERS}b-v1.5"
# Hardcoded path: edit this path for your setup.
CACHE_DIR = os.path.expanduser("~/scratch/huggingface/hub/")
PRETRAINED_PROJECTOR_NAME_ORIGINAL = (
    f"liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-{NUM_PARAMETERS}b-v1.5"
)
# Hardcoded path: edit this path for your setup.
PRETRAINED_PROJECTOR_LBASILE_PATH = os.path.expanduser(f"~/scratch/pretrained_projectors/mm_projector_lbasile-llava-v1.5-{NUM_PARAMETERS}B.bin")

def replace_lm_and_projector(
    projector_model_name_or_path: str,
    target_repo_id: str,
    projector_source: str = PRETRAINED_PROJECTOR_NAME_ORIGINAL,
):
    try:
        model, processor = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=BASE_MODEL_NAME,
            cache_dir=CACHE_DIR, 
            device_map="cpu",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        
        model = replace_multimodal_projector(
            model,
            projector_model_name_or_path,
            CACHE_DIR,
        )
        
        model = replace_multimodal_lm(
            model,
            LANGUAGE_MODEL_NAME,
            CACHE_DIR,
            dtype=torch.bfloat16,
        )
        
        print(f"  Pushing to Hub: {target_repo_id}...")
        create_repo(target_repo_id, exist_ok=True, private=False)

        readme_content = (
        f"""---
        tags:
        - image-text-to-text
        - llava
        - multimodal
        - vision-language
        ---

        # {target_repo_id.split('/')[-1]}

        LLaVA v1.5 {NUM_PARAMETERS}B model with Vicuna 1.5 LM and {projector_source} pretrained projector.
        """
        )

        model.push_to_hub(
            target_repo_id,
            commit_message="Port LM and pretrained projector weights."
        )
        processor.push_to_hub(target_repo_id)
        
        # Upload the generated README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=target_repo_id,
            repo_type="model",
        )
        
        print(f"  \033[92m✓ Successfully pushed to {target_repo_id}\033[0m")
        
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"  \033[91m✗ Error: {e}\033[0m")
        import traceback
        traceback.print_exc()
        return False
        

if __name__ == "__main__":
    # Create pretrained model with lbasile projector
    print(F"Setting up LLaVA v1.5 {NUM_PARAMETERS}B pretrained model with lbasile projector...")
    replace_lm_and_projector(
        projector_model_name_or_path=PRETRAINED_PROJECTOR_LBASILE_PATH,
        target_repo_id=f"l11p/llava-v1.5-{NUM_PARAMETERS}b-pretrained-lbasile",
        projector_source="custom training by lbasile",
    )
    
    # # Create pretrained model with original projector
    # print(f"Setting up LLaVA v1.5 {NUM_PARAMETERS}B pretrained model with original projector...")
    # replace_lm_and_projector(
    #     projector_model_name_or_path=PRETRAINED_PROJECTOR_NAME_ORIGINAL,
    #     target_repo_id=f"l11p/llava-v1.5-{NUM_PARAMETERS}b-pretrained-liuhaotian",
    #     projector_source="original pretrained projector",
    # )
    
