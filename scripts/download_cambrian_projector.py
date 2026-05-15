"""Download mm_projector.bin from cambrian-8b-pretrain and inspect its contents."""

from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch
import os

REPO_ID = "nyu-visionx/cambrian-8b-pretrain"
FILENAME = "mm_projector.bin"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cambrian")

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"Downloading {FILENAME} from {REPO_ID} ...")
path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=LOCAL_DIR)
print(f"Saved to: {path}\n")

state_dict = torch.load(path, map_location="cpu", weights_only=True)

print(f"Type: {type(state_dict)}")
print(f"Number of keys: {len(state_dict)}\n")

print("=" * 80)
print(f"{'Key':<60} {'Shape':<25} {'Dtype'}")
print("=" * 80)
total_params = 0
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
        print(f"{k:<60} {str(tuple(v.shape)):<25} {v.dtype}")
        total_params += v.numel()
    else:
        print(f"{k:<60} {str(type(v)):<25}")

print("=" * 80)
print(f"\nTotal parameters: {total_params:,}")
print(f"Total size (float32): {total_params * 4 / 1024**2:.1f} MB")
