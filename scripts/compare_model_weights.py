"""
Compare weights layer by layer between two models:
  - lbasile/LLaVA-OneVision-1.5-8B_7_25
  - lmms-lab/LLaVA-OneVision-1.5-8B-Base
"""

import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict


MODEL_A = "lbasile/LLaVA-OneVision-1.5-8B_7_25"
MODEL_B = "lmms-lab/LLaVA-OneVision-1.5-8B-Base"


def load_state_dict(model_id):
    print(f"Loading {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    return model.state_dict()


def compare_state_dicts(sd_a, sd_b):
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    only_in_a = keys_a - keys_b
    only_in_b = keys_b - keys_a
    common = keys_a & keys_b

    if only_in_a:
        print(f"\n[Keys only in {MODEL_A}]")
        for k in sorted(only_in_a):
            print(f"  {k}")

    if only_in_b:
        print(f"\n[Keys only in {MODEL_B}]")
        for k in sorted(only_in_b):
            print(f"  {k}")

    print(f"\nComparing {len(common)} common keys ...\n")

    equal_keys = []
    diff_keys = []
    shape_mismatch_keys = []

    for key in sorted(common):
        t_a = sd_a[key]
        t_b = sd_b[key]

        if t_a.shape != t_b.shape:
            shape_mismatch_keys.append((key, t_a.shape, t_b.shape))
            continue

        if torch.equal(t_a, t_b):
            equal_keys.append(key)
        else:
            max_diff = (t_a.float() - t_b.float()).abs().max().item()
            mean_diff = (t_a.float() - t_b.float()).abs().mean().item()
            diff_keys.append((key, max_diff, mean_diff))

    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Total common keys   : {len(common)}")
    print(f"  Shape mismatches    : {len(shape_mismatch_keys)}")
    print(f"  Exactly equal       : {len(equal_keys)}")
    print(f"  Different           : {len(diff_keys)}")
    print()

    if shape_mismatch_keys:
        print(f"[Shape mismatches]")
        for key, sa, sb in shape_mismatch_keys:
            print(f"  {key:80s}  {str(sa):20s} vs {str(sb)}")
        print()

    if diff_keys:
        print(f"[Different layers] (sorted by max absolute difference)")
        diff_keys.sort(key=lambda x: -x[1])
        print(f"  {'Layer':<80s} {'Max |diff|':>12} {'Mean |diff|':>12}")
        print(f"  {'-'*80} {'-'*12} {'-'*12}")
        for key, max_diff, mean_diff in diff_keys:
            print(f"  {key:<80s} {max_diff:>12.6e} {mean_diff:>12.6e}")
        print()

    if equal_keys:
        print(f"[Equal layers]")
        for key in equal_keys:
            print(f"  {key}")
        print()


if __name__ == "__main__":
    sd_a = load_state_dict(MODEL_A)
    sd_b = load_state_dict(MODEL_B)
    compare_state_dicts(sd_a, sd_b)
