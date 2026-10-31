"""Compare in-house VLM benchmark scores against the public lmarena vision
leaderboard. Requires a local ./vlm_eval_kit.json (not part of this repo,
not fetched automatically) with the same {model: {task: {field: score}}}
shape as the leaderboard dataset below."""

import json

from datasets import load_dataset

# Load latest vision leaderboard
ds = load_dataset("lmarena-ai/leaderboard-dataset", "vision", split="latest")

with open("./vlm_eval_kit.json", "r") as f:
    data = json.load(f)

model_names = {
    "Cambrian-8B",
    "InternVL2-8B",
    "LLaVA-v1.5-13B",
    "LLaVA-v1.5-7B",
    "Llama-3.2-11B-Vision-Instruct",
    "Llama-3.2-90B-Vision-Instruct",
}

tasks = (
    "SEEDBench_IMG",
    "MME",
    "MMMU_VAL",
    "AI2D",
    "COCO_VAL",
    "ScienceQA_VAL",
    "ScienceQA_TEST",
    "POPE",
    "BLINK",
)
# SEEDBench2_Plus

results = {name: {} for name in model_names}

for model in model_names:
    for task in tasks:
        field = "Overall"
        if "coco_val" == task.lower():
            field = "CIDEr"
        res = data["results"][model][task]
        results[model][task] = res[field]

data["results"]["InternVL2-8B"].keys()
results


datasets = {
    "gqa",
    "vqav2",
    "vizwiz",
    "textvqa",
    "ai2d",
    "scienceqa",
    "mmbench-en",
    "mmbench-cn",
    "mme",
    "pope",
    "seed",
    "mmmu",
    "blink",
    "mmvp",
    "naturalbench",
    "vlmsareblind",
    "cvbench",
    "coco",
    "flickr30k",
}


data["results"]["Llama-3.2-90B-Vision-Instruct"]["LLaVABench"]
