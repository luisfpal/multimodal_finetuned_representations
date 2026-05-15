"""
Build a 'pretrained' Cambrian-8B checkpoint by assembling:
  - LLM: NousResearch/Meta-Llama-3-8B-Instruct
  - Vision encoders: SigLIP, CLIP, DINOv2, ConvNeXt-CLIP (same as cambrian-8b)
  - Connector (SVA projector): nyu-visionx/cambrian-8b-pretrain  (mm_projector.bin)

The output is saved locally and pushed to HuggingFace as lbasile/cambrian-8b-pretrained,
with the same folder structure as nyu-visionx/cambrian-8b.
"""

import sys
import os
import json
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download

# Add cambrian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cambrian'))

from cambrian.model.language_model.cambrian_llama import (
    CambrianConfig,
    CambrianLlamaForCausalLM,
)
from transformers import AutoTokenizer


def main():
    output_dir = Path("/orfeo/cephfs/scratch/dssc/lbasile/plm/results/cambrian-8b-pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Build the config (identical to nyu-visionx/cambrian-8b)
    # ------------------------------------------------------------------ #
    print("==> Building CambrianConfig ...")
    config = CambrianConfig(
        # Llama-3-8B architecture
        _name_or_path="./Meta-Llama-3-8B-Instruct",
        architectures=["CambrianLlamaForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=128000,
        eos_token_id=128001,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=14336,
        max_position_embeddings=8192,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        rope_theta=500000.0,
        tie_word_embeddings=False,
        use_cache=True,
        vocab_size=128256,
        # Cambrian vision config
        mm_vision_tower_aux_list=[
            "siglip/CLIP-ViT-SO400M-14-384",
            "openai/clip-vit-large-patch14-336",
            "facebook/dinov2-giant-res378",
            "clip-convnext-XXL-multi-stage",
        ],
        mm_vision_tower_aux_token_len_list=[576, 576, 576, 9216],
        vision_tower_aux_token_len_list=[576, 576, 576, 9216],
        mm_projector_type="sva",
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        use_mm_proj=True,
        vision_hidden_size=1024,
        image_token_len=576,
        image_position=91,
        image_aspect_ratio="pad",
        mm_patch_merge_type="flat",
        connector_depth=3,
        connector_only=False,
        num_query_group=1,
        query_num_list=[576],
        num_of_vision_sampler_layers=10,
        start_of_vision_sampler_layers=0,
        stride_of_vision_sampler_layers=3,
        freeze_mm_mlp_adapter=False,
        tune_mm_mlp_adapter=False,
        unfreeze_mm_vision_tower=False,
        mm_projector_lr=None,
        mm_vision_sampler_lr=None,
        mm_vision_tower_lr=None,
        tokenizer_model_max_length=2048,
        tokenizer_padding_side="right",
    )

    # ------------------------------------------------------------------ #
    # 2. Instantiate the model (random connector weights, random LLM)
    # ------------------------------------------------------------------ #
    print("==> Instantiating CambrianLlamaForCausalLM (with random weights) ...")
    model = CambrianLlamaForCausalLM(config)

    # ------------------------------------------------------------------ #
    # 3. Load LLM weights from Meta-Llama-3-8B-Instruct
    # ------------------------------------------------------------------ #
    print("==> Loading Llama-3-8B-Instruct LLM weights ...")
    from transformers import LlamaForCausalLM
    llm = LlamaForCausalLM.from_pretrained(
        "NousResearch/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Copy LLM weights into the Cambrian model
    # The Cambrian model's LLM backbone lives at model.model (CambrianLlamaModel inherits LlamaModel)
    # We need to copy: embed_tokens, layers, norm, lm_head
    llm_state = llm.state_dict()
    cambrian_state = model.state_dict()

    loaded_keys = []
    for key, value in llm_state.items():
        if key in cambrian_state:
            cambrian_state[key] = value
            loaded_keys.append(key)

    print(f"   Loaded {len(loaded_keys)} LLM weight tensors")
    model.load_state_dict(cambrian_state, strict=False)
    del llm, llm_state
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------ #
    # 4. Load vision towers
    # ------------------------------------------------------------------ #
    print("==> Loading vision towers ...")
    vision_tower_aux_list = model.get_model().get_vision_tower_aux_list()
    for i, vt in enumerate(vision_tower_aux_list):
        print(f"   Loading vision tower {i}: {vt.vision_tower_name} ...")
        vt.load_model()
        vt.to(dtype=torch.bfloat16)
        print(f"   Vision tower {i} loaded. hidden_size={vt.hidden_size}")

    # ------------------------------------------------------------------ #
    # 5. Load pretrained connector (mm_projector.bin)
    # ------------------------------------------------------------------ #
    print("==> Downloading pretrained connector from nyu-visionx/cambrian-8b-pretrain ...")
    projector_path = hf_hub_download(
        repo_id="nyu-visionx/cambrian-8b-pretrain",
        filename="mm_projector.bin",
    )
    print(f"   Downloaded to: {projector_path}")

    mm_projector_weights = torch.load(projector_path, map_location="cpu", weights_only=False)
    print(f"   Projector checkpoint has {len(mm_projector_weights)} keys")

    # Use the same loading logic as cambrian_arch.py initialize_vision_modules
    def get_w(weights, keyword):
        return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword + '.' in k}

    # mm_projector (the final Linear->GELU->Linear)
    model.get_model().mm_projector.load_state_dict(
        get_w(mm_projector_weights, 'mm_projector'), strict=True
    )
    print("   Loaded mm_projector weights")

    # mm_projector_aux_0..3  (per-tower projectors)
    for aux_i in range(len(vision_tower_aux_list)):
        getattr(model.get_model(), f'mm_projector_aux_{aux_i}').load_state_dict(
            get_w(mm_projector_weights, f'mm_projector_aux_{aux_i}'), strict=True
        )
        print(f"   Loaded mm_projector_aux_{aux_i} weights")

    # vision_sampler_0 (the pre-LLM connector sampler, num_query_group=1)
    num_query_group = config.num_query_group
    for query_group_i in range(num_query_group):
        getattr(model.get_model(), f"vision_sampler_{query_group_i}").load_state_dict(
            get_w(mm_projector_weights, f"vision_sampler_{query_group_i}"), strict=True
        )
        print(f"   Loaded vision_sampler_{query_group_i} weights")

    # vision_sampler_layers (the 10 in-LLM sampler layers)
    model.get_model().vision_sampler_layers.load_state_dict(
        get_w(mm_projector_weights, 'vision_sampler_layers'), strict=True
    )
    print("   Loaded vision_sampler_layers weights")

    # vision_query
    model.get_model().vision_query.data = mm_projector_weights['model.vision_query']
    print("   Loaded vision_query")

    # image_newline
    model.get_model().image_newline.data = mm_projector_weights['model.image_newline']
    print("   Loaded image_newline")

    del mm_projector_weights
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------ #
    # 6. Cast model to float16 for storage
    # ------------------------------------------------------------------ #
    print("==> Casting model to bfloat16 ...")
    model = model.to(dtype=torch.bfloat16)

    # ------------------------------------------------------------------ #
    # 7. Save tokenizer
    # ------------------------------------------------------------------ #
    print("==> Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Meta-Llama-3-8B-Instruct", use_fast=False
    )
    tokenizer.save_pretrained(output_dir)

    # ------------------------------------------------------------------ #
    # 8. Save model (sharded safetensors, matching cambrian-8b structure)
    # ------------------------------------------------------------------ #
    print("==> Saving model ...")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # ------------------------------------------------------------------ #
    # 9. Save generation_config.json
    # ------------------------------------------------------------------ #
    gen_config = {
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "transformers_version": "4.40.1",
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)
    print("   Saved generation_config.json")

    print(f"\n==> Checkpoint saved to: {output_dir}")
    print("   Files:")
    for p in sorted(output_dir.iterdir()):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"   {p.name:50s} {size_mb:8.1f} MB")

    # ------------------------------------------------------------------ #
    # 10. Push to HuggingFace
    # ------------------------------------------------------------------ #
    print("\n==> Pushing to HuggingFace as lbasile/cambrian-8b-pretrained ...")
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo("lbasile/cambrian-8b-pretrained", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id="lbasile/cambrian-8b-pretrained",
        commit_message="Upload pretrained Cambrian-8B checkpoint (Llama-3-8B-Instruct + frozen vision encoders + pretrained SVA connector)",
    )
    print("==> Done! Model uploaded to https://huggingface.co/lbasile/cambrian-8b-pretrained")


if __name__ == "__main__":
    main()
