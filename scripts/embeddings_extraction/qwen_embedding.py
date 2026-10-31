import argparse
import json
import os

# from utils.create_datasets import TASK_DICT
# from models.modules import (
#     add_pooling_layers,
#     last_token_pool,
#     mean_pool,
#     instruction_template_qwen3,
# )
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer

from utils.embeddings_extraction.embedding_utils import (
    EvalContext,
    add_pooling_layers,
    encode_dataset,
    instruction_template_embeddinggemma,
    instruction_template_qwen3,
    last_token_pool,
    make_collate_fn,
    mean_pool,
    prepare_text_dataset,
)


def prepare_coco_icqa(max_num_samples):
    dataset = load_dataset("l11p/COCOQA-restval-ICQA")["train"]
    num_samples = len(dataset)

    def return_caption(strings):
        return ". ".join(s.strip().rstrip(".") for s in strings) + "."

    max_num_samples = max_num_samples
    if max_num_samples is None:
        max_num_samples = num_samples
    all_sentences = [return_caption(cap) for cap in dataset["captions"]][
        :max_num_samples
    ]
    unique_texts, text_to_idx = [], {}
    for text in all_sentences:
        h = hash(text)
        if h not in text_to_idx:
            text_to_idx[h] = len(unique_texts)
            unique_texts.append(text)

    indices1 = [text_to_idx[hash(s)] for s in all_sentences]
    n_dedup = len(all_sentences) - len(unique_texts)

    return unique_texts, indices1, n_dedup


def prepare_detailed_descriptions(max_num_samples):

    with open("./detail_23k.json", "r") as f:
        descriptions = json.load(f)

    desc = {}
    for d in descriptions:
        desc[d["image"]] = d["conversations"][1]["value"]

    return list(desc.values()), list(desc.keys()), 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument(
        "--filename",
        type=str,
        default="",
        help="Path to save the output .pt file containing 'embeddings' and 'indices'.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/embedding_similarity",
    )
    parser.add_argument(
        "--similar_prompts_json",
        type=str,
        default="first50_similar_prompts.json",
        help="Filename under --out_dir: first 50 unique prompts and top-10 neighbors by cosine similarity.",
    )
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=None,
        help="Filename under --out_dir: first 50 unique prompts and top-10 neighbors by cosine similarity.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cocoqa",
        help="Filename under --out_dir: first 50 unique prompts and top-10 neighbors by cosine similarity.",
    )
    args = parser.parse_args()
    return args


@torch.inference_mode()
def evaluate_one_sts(texts_ds, model, batch_size, eval_context: EvalContext):
    model.eval()

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )
    embeddings = encode_dataset(model, texts_ds, batch_size, collate_fn)
    return embeddings


def first_prompts_topk_neighbors_json(
    embeddings: torch.Tensor,
    unique_texts: list,
    n_prompts: int = 50,
    k: int = 10,
) -> list:
    """Cosine similarity in embedding space; excludes self from neighbors."""
    n_unique = embeddings.shape[0]
    n_q = min(n_prompts, n_unique)
    k_eff = min(k, max(0, n_unique - 1))
    if k_eff == 0:
        return [
            {"index": i, "prompt": unique_texts[i], "neighbors": []} for i in range(n_q)
        ]

    emb = F.normalize(embeddings.float(), dim=1)
    # [n_q, N] only — avoids O(N^2) full similarity matrix
    sim = emb[:n_q] @ emb.T

    out = []
    for i in range(n_q):
        row = sim[i].clone()
        row[i] = -float("inf")
        vals, idx = torch.topk(row, k=k_eff)
        neighbors = [
            {
                "index": int(j),
                "prompt": unique_texts[int(j)],
                "cosine_similarity": float(v),
            }
            for j, v in zip(idx.tolist(), vals.tolist())
        ]
        out.append({"index": i, "prompt": unique_texts[i], "neighbors": neighbors})
    return out


def main():
    args = parse_args()

    dist.init_process_group(
        "nccl",
        device_id=torch.device("cuda", LOCAL_RANK),
        timeout=timedelta(seconds=60),
    )
    rank = dist.get_rank()

    torch.cuda.set_device(LOCAL_RANK)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, trust_remote_code=True
    )

    if "qwen3" in args.model_name_or_path.lower():
        model_name = "qwen3"
        if "0.6" in args.model_name_or_path.lower():
            model_name += "_0.6b"
        if "8" in args.model_name_or_path.lower():
            model_name += "_8b"

        instruction_template = instruction_template_qwen3
        pool_fn = last_token_pool
        add_special_tokens = False
        eot_id = tokenizer.pad_token_id
    elif "embeddinggemma" in args.model_name_or_path.lower():
        model_name = "embeddinggemma"
        instruction_template = instruction_template_embeddinggemma
        pool_fn = mean_pool
        add_special_tokens = True
        eot_id = None
    else:
        raise ValueError(
            f"Unrecognized model '{args.model_name_or_path}'. "
            "Expected a path containing 'qwen3' or 'embeddinggemma'."
        )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to("cuda")
    model = add_pooling_layers(model, pool_fn=pool_fn)

    model = DDP(model, device_ids=[LOCAL_RANK])
    model = torch.compile(model)

    eval_context = EvalContext(
        tokenizer=tokenizer,
        padding_side="right",
        eot_id=eot_id,
        add_special_tokens=add_special_tokens,
        world_size=WORLD_SIZE,
        rank=RANK,
    )

    if args.dataset_name == "llava_descriptions":
        unique_texts, indices1, n_dedup = prepare_detailed_descriptions(
            args.max_num_samples
        )
    elif args.dataset_name == "cocoqa":
        unique_texts, indices1, n_dedup = prepare_coco_icqa(args.max_num_samples)
    else:
        raise ValueError("wrong dataset name")

    task_metadata = ""
    texts_ds = prepare_text_dataset(
        unique_texts, task_metadata, instruction_template, tokenizer, rank
    )

    embeddings = evaluate_one_sts(texts_ds, model, 32, eval_context)
    # Keep on CPU as float32; shape: [n_unique, embed_dim]
    embeddings_cpu = embeddings.cpu()
    # LongTensor of shape [n_all]: maps each original caption to its row in embeddings_cpu
    # indices_tensor = torch.tensor(indices1, dtype=torch.long)

    if rank == 0:
        print(
            f"Encoded {len(unique_texts)} unique texts ({n_dedup} duplicates removed)"
        )
        print(f"embeddings shape : {embeddings_cpu.shape}")
        # print(f"indices shape    : {indices_tensor.shape}")

        filename = f"{model_name}_{args.dataset_name}"

        if args.filename != "":
            filename += f"_{args.filename}"

        max_num_samples = ""
        if args.max_num_samples is not None:
            max_num_samples = f"_{args.max_num_samples}"

        out_path = f"{args.out_dir}/{filename}{max_num_samples}.pt"  # os.path.join(args.out_dir, args.filename)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(
            {
                "embeddings": embeddings_cpu,  # [n_unique, embed_dim]
                # "indices": indices_tensor,  # [n_all]  caption_i -> embeddings[indices[i]]
                "indices": indices1,
            },
            out_path,
        )
        print(f"Saved embeddings and indices to {out_path}")

        sim_path = os.path.join(args.out_dir, args.similar_prompts_json)
        report = first_prompts_topk_neighbors_json(
            embeddings_cpu, unique_texts, n_prompts=20, k=10
        )
        with open(sim_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved first-50 similar-prompts report to {sim_path}")

    dist.destroy_process_group()


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
