"""Logit-lens evaluation: per-layer answer accuracy obtained by decoding the residual stream on VLM benchmarks.

Inputs: --results-dir, --model-name-or-path, --batch-size, --seed.
Outputs: safetensors files with per-layer accuracies under --results-dir.
"""

import argparse
import gc
import os
import sys
import traceback

import numpy as np
import torch
from dotenv import load_dotenv
from safetensors.numpy import save_file
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature

from src.residual_stream_tracer import residual_stream_tracer
from utils.cambrian_utils import is_cambrian_model
from utils.constants import (
    AI2D_ANSWER_CHOICES,
    MMBENCH_ANSWER_CHOICES,
    MME_ANSWER_CHOICES,
    MMSTAR_ANSWER_CHOICES,
    POPE_ANSWER_CHOICES,
    SQA_ANSWER_CHOICES,
)

from utils import (
    get_dataloader,
    load_hf_model_and_processor_or_tokenizer,
    seed_all,
)
from utils.internvl_utils import (
    IMG_CONTEXT_TOKEN,
    generate_responses,
    get_combined_embeddings,
)

# Load environment variables
load_dotenv()


# Configuration
MODEL_DTYPE = torch.bfloat16
ATTN_IMPLEMENTATION = "sdpa"
RESIDUAL_STREAM_TYPE = "output_layer"
TOKENS_POOLING_METHOD = "last"
DEFAULT_TOP_K_LOGITS = 10

# Dataset Configuration
DATASETS = {
    "l11p/COCOQA-restval-ICQA": {
        "guide_text": "Answer the question using a single word or phrase.",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "train",
        "type": "open_ended",
        "remove_yes_no": True,
        "eval_methods": ["logit_lens"],
    },
    "l11p/VQAv2-COCO-restval-ICQ": {
        "guide_text": "Answer the question using a single word or phrase.",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "train",
        "type": "open_ended",
        "remove_yes_no": True,
        "eval_methods": ["logit_lens"],
    },
    "l11p/MME-test-captioned": {
        "guide_text": "Answer the question using a single word or phrase.",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "train",
        "type": "multiple_choice",
        "choices": MME_ANSWER_CHOICES,  # Yes, No
        "eval_methods": ["linear_probe", "logit_lens"],
    },
    "l11p/MMBench_EN-dev-captioned": {
        "guide_text": "Answer with the option's letter from the given choices directly",
        "downsample_size": 2000,
        "filter_mmbench_4_choices": True,
        "subsets": ["images_qa"],
        "split": "train",
        "type": "multiple_choice",
        "choices": MMBENCH_ANSWER_CHOICES,  # A, B, C, D
        "eval_methods": ["linear_probe", "logit_lens"],
    },
    "lmms-lab/ai2d": {
        "guide_text": "Answer with the option's letter from the given choices directly",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "test",
        "type": "multiple_choice",
        "choices": AI2D_ANSWER_CHOICES,  # A, B, C, D
        "eval_methods": ["linear_probe", "logit_lens"],
    },
    "lmms-lab/POPE": {
        "guide_text": "Answer the question using a single word or phrase.",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "test",
        "type": "multiple_choice",
        "choices": POPE_ANSWER_CHOICES,  # Yes, No
        "eval_methods": ["linear_probe", "logit_lens"],
    },
    "l11p/GQA-val-balanced-noyesno-captioned": {
        "guide_text": "Answer the question using a single word or phrase.",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "train",  # It was created from 'val' but usually loaded as 'train' if from Hub unless specified
        "type": "open_ended",
        "eval_methods": ["logit_lens"],
        # No exclude_yes_no needed as dataset is already filtered
    },
    "lmms-lab/textvqa": {
        "guide_text": "Answer the question using a single word or phrase.",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "validation",
        "type": "open_ended",
        "eval_methods": ["logit_lens"],
    },
    "lmms-lab/ScienceQA": {
        "guide_text": "Answer with the option's letter from the given choices directly",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "test",
        "type": "multiple_choice",
        "choices": SQA_ANSWER_CHOICES,  # A, B, C, D, E
        "eval_methods": ["linear_probe", "logit_lens"],
    },
    "Lin-Chen/MMStar": {
        "guide_text": "Answer with the option's letter from the given choices directly",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "val",
        "type": "multiple_choice",
        "choices": MMSTAR_ANSWER_CHOICES,  # A, B, C, D
        "eval_methods": ["linear_probe", "logit_lens"],
        "open_ended_prompt": True,  # options already embedded in question text
    },
    "vikhyatk/CountBenchQA": {
        "guide_text": "Answer with just a number.",
        "downsample_size": 2000,
        "subsets": ["images_qa"],
        "split": "test",
        "type": "open_ended",
        "eval_methods": ["logit_lens"],
    },
}


def apply_rmsnorm(x, second_moment, weight, variance_eps):
    print(weight.shape)
    # second_moment = second_moment.unsqueeze(-1)
    print(second_moment.shape, x.shape)
    x = x * torch.rsqrt(second_moment + variance_eps)
    print(x.shape)
    return weight * x


def apply_rmsnorm_residual(
    x: torch.Tensor,  # Shape: (n, t, u, d)
    second_moment: torch.Tensor,  # Shape: (n, t)
    weight: torch.Tensor,  # Shape: (d,)
    variance_eps: float,
):
    # assert (
    #     x.dim() == 3
    # ), f"Input tensor x must have 3 dimensions (n, r, d). Found {x.shape}"
    assert second_moment.dim() == 2, (
        f"second moments must have 2 dimensions (n, t). Found {second_moment.shape}"
    )
    assert weight.dim() == 1, f"weight must have 1 dimension (d,). Found {weight.shape}"
    assert x.size(0) == second_moment.size(0), (
        f"n dimension mismatch. {x.size(0)=} {second_moment.size(0)=}"
    )
    assert x.size(-1) == weight.size(0), (
        f"d dimension mismatch. {x.size(-1)=} {weight.size(0)=}"
    )

    squeeze_tokens = False
    if x.ndim == 2:
        # assume the token dimension is 1
        x = x.view(x.size(0), 1, x.size(1))
        squeeze_tokens = True

    # Calculate the number of middle dimensions in x (n, t, ..., d)
    num_unit_dims = x.ndim - 3

    # Reshape mean and std to broadcast with x
    second_moment = second_moment.view(
        second_moment.shape[0], second_moment.shape[1], *[1] * num_unit_dims, 1
    )  # Shape: (n, t, ..., 1)

    # Reshape gamma and beta to broadcast with x
    weight = weight.view(*[1] * (num_unit_dims + 1), -1)  # Shape: (1, ..., d)

    x = x * torch.rsqrt(second_moment + variance_eps)

    # CRITICAL: Cast back to weight's dtype (bfloat16) BEFORE multiplication
    # to match LlamaRMSNorm behavior
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)

    out = weight * x

    if squeeze_tokens:
        out = out.view(out.size(0), out.size(2))

    return out


def get_target_token_ids(tokenizer, choices):
    """
    Get token IDs for the answer choices.
    """
    ids = []
    for c in choices:
        # We assume the model answers with just the letter/word
        # LLaVA tokenizer behavior with 'A' vs ' A' can be tricky,
        # but usually for generation it's the token itself.
        # We use the last token ID if multiple are returned, but these should be single tokens.
        tokenized = tokenizer.encode(c, add_special_tokens=False)
        ids.append(tokenized[0])
    return ids


def evaluate_logit_lens(
    model,
    tokenizer,
    residual_stream,
    labels,
    dataset_config,
    dataloader=None,
    top_k_override=None,
):
    """
    Evaluate logit lens accuracy.
    """
    num_layers, num_samples, _ = residual_stream.shape
    layer_hits = torch.zeros(num_layers, device="cpu")

    dataset_type = dataset_config.get("type", "open_ended")

    # Prepare for Multiple Choice if needed
    target_ids = None
    target_ids_tensor = None
    choices = None

    # Storage for balanced accuracy calculation
    all_ground_truths = []
    all_predictions = [[] for _ in range(num_layers)]

    if dataset_type == "multiple_choice":
        choices = dataset_config["choices"]
        target_ids = get_target_token_ids(tokenizer, choices)
        target_ids_tensor = torch.tensor(target_ids, device=model.device)
        print(f"  🎯 Multiple Choice Targets: {choices} -> IDs: {target_ids}")

    print(f"  📊 Evaluating ({dataset_type})...")

    # We iterate over samples
    # Using batch processing could be faster but residual stream is already computed per sample usually or loaded as tensor
    # residual_stream is (layers, samples, hidden).

    # Helper iterator
    def sample_iterator():
        if dataloader:
            global_idx = 0
            for batch in tqdm(dataloader, desc="Evaluating Samples"):
                print(len(batch))
                inputs = batch[0]
                batch_labels = batch[-1]
                pixel_values = None
                if "intern" in model.__class__.__name__.lower():
                    pixel_values = batch[1]
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs = inputs[0]
                if hasattr(inputs, "to"):
                    inputs = inputs.to(model.device)

                curr_bs = len(batch_labels)
                for b in range(curr_bs):
                    yield global_idx + b, inputs, b, batch_labels[b], pixel_values
                global_idx += curr_bs
        else:
            for i in tqdm(range(num_samples), desc="Evaluating Samples"):
                yield i, None, 0, labels[i], None

    # InternVL-specific: get image context token id for combined embeddings
    img_context_token_id = None
    if "intern" in model.__class__.__name__.lower():
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required for InternVL models to get image context token ID."
            )
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    for i, inputs_batch, b_idx, label_text, pixel_values_batch in sample_iterator():
        ground_truth = label_text.strip()

        # sample_res: (num_layers, hidden_size)
        sample_res = residual_stream[:, i, :].to(model.device).to(MODEL_DTYPE)
        sample_res = sample_res.unsqueeze(0)
        sample_res = sample_res

        with torch.no_grad():
            second_moment = (
                sample_res[:, -1].to(torch.float32).pow(2).mean(dim=-1).unsqueeze(1)
            )

            if "llava" in model.__class__.__name__.lower():
                normalized_res = (
                    apply_rmsnorm_residual(
                        sample_res.to(torch.float32),
                        second_moment,
                        model.model.language_model.norm.weight,
                        model.model.language_model.norm.variance_epsilon,
                    )
                    .squeeze()
                    .to(MODEL_DTYPE)
                )

                # logits: (num_layers, vocab_size)
                logits = model.lm_head(normalized_res)

            elif "intern" in model.__class__.__name__.lower():
                normalized_res = (
                    apply_rmsnorm_residual(
                        sample_res.to(torch.float32),
                        second_moment,
                        model.language_model.model.norm.weight,
                        model.language_model.model.norm.variance_epsilon,
                    )
                    .squeeze()
                    .to(MODEL_DTYPE)
                )

                # logits: (num_layers, vocab_size)
                logits = model.language_model.output(normalized_res)

            elif is_cambrian_model(model=model):
                normalized_res = (
                    apply_rmsnorm_residual(
                        sample_res.to(torch.float32),
                        second_moment,
                        model.model.norm.weight,
                        model.model.norm.variance_epsilon,
                    )
                    .squeeze()
                    .to(MODEL_DTYPE)
                )

                # logits: (num_layers, vocab_size)
                logits = model.lm_head(normalized_res.to(model.lm_head.weight.dtype))
            else:
                raise ValueError(
                    "Unsupported model architecture for logit-lens evaluation: "
                    f"{model.__class__.__name__}"
                )

            # --- Added Logic: Print Top 10 tokens and Generate ---
            try:
                top_k_print = 10
                _, ids_debug = logits.topk(top_k_print, dim=-1)
                decoded_debug = tokenizer.batch_decode(
                    ids_debug.view(-1, 1), skip_special_tokens=True
                )
                decoded_layers_debug = [
                    decoded_debug[j : j + top_k_print]
                    for j in range(0, len(decoded_debug), top_k_print)
                ]

                if i < 5:
                    print(f"\nSample {i}: Ground Truth: {ground_truth}")
                    for layer_idx, tokens in enumerate(decoded_layers_debug):
                        print(f"  Layer {layer_idx}: {tokens}")

                    # if inputs_batch is not None:
                    #     # Slice for this sample
                    #     sample_input = {}
                    #     # print("DEBUG")
                    #     # print(inputs_batch)
                    #     # print(f"Batch index: {b_idx}")
                    #     # print(f"Batch size: {len(inputs_batch)}")
                    #     for k, v in inputs_batch.items():
                    #         if isinstance(v, torch.Tensor):
                    #             sample_input[k] = v[b_idx : b_idx + 1]
                    #         else:
                    #             sample_input[k] = v[b_idx : b_idx + 1]
                    #     # print("Sample input for generation:")
                    #     # print(sample_input)

                    #     if (
                    #         pixel_values_batch is not None
                    #         and "intern" in model.__class__.__name__.lower()
                    #     ):
                    #         # print("  🖼️ Processing image context for generation...")
                    #         combined_embeds = get_combined_embeddings(
                    #             model,
                    #             input_ids=sample_input["input_ids"],
                    #             pixel_values=pixel_values_batch[b_idx : b_idx + 1],
                    #             img_context_token_id=img_context_token_id,
                    #         )
                    #         sample_input = BatchFeature(
                    #             {
                    #                 "inputs_embeds": combined_embeds,
                    #                 "attention_mask": sample_input["attention_mask"],
                    #             }
                    #         )

                    #     # Generate
                    #     if "llava" in model.__class__.__name__.lower():
                    #         gen_out = model.generate(**sample_input, max_new_tokens=20)
                    #         gen_out = gen_out[:, sample_input["input_ids"].shape[1] :]
                    #         gen_text = tokenizer.decode(
                    #             gen_out[0], skip_special_tokens=True
                    #         )
                    #     elif "intern" in model.__class__.__name__.lower():
                    #         gen_text = generate_responses(
                    #             model=model,
                    #             tokenizer=tokenizer,
                    #             inputs_embeds=sample_input["inputs_embeds"],
                    #             attention_mask=sample_input["attention_mask"],
                    #             max_new_tokens=20,
                    #         )[0]
                    #     print(f"  🤖 Model Generation: {gen_text}")
            except Exception as e:
                print(f"  ⚠️ Debug Print Failed: {e}")
                traceback.print_exc()
            # ----------------------------------------------------

            # sys.exit(0)

            if dataset_type == "multiple_choice":
                # Restrict logits to target tokens
                # choice_logits: (num_layers, num_choices)
                choice_logits = logits[:, target_ids_tensor]

                # Get prediction (index in target_ids)
                pred_indices = torch.argmax(choice_logits, dim=-1)

                # Check against ground truth
                # Ground truth is a string (e.g., "A", "Yes")
                # We need to find its index in choices
                try:
                    gt_idx = choices.index(ground_truth)

                    # Store ground truth and predictions for balanced accuracy
                    all_ground_truths.append(gt_idx)
                    for l_idx in range(num_layers):
                        all_predictions[l_idx].append(pred_indices[l_idx].item())

                except ValueError:
                    # Ground truth not in choices (should not happen usually)
                    pass

            else:  # Open Ended
                top_k = top_k_override if top_k_override else DEFAULT_TOP_K_LOGITS

                # Get top-k tokens
                topk_vals, topk_ids = logits.topk(top_k, dim=-1)

                # Get the initial token of the answer
                # We tokenize the ground truth (assumed to be text) and take the first token
                # Note: This is sensitive to how the answer is formatted in dataset and tokenizer behavior (e.g. leading space)
                # But requested behavior is "one initial token".

                # Check with and without leading space just in case, but prioritize exact match?
                # User instructions: "check the first token of the capitalized and lowercase version of the answer. it is correct if the model returns any of the two tokens"

                target_token_ids = []

                # Check lowercase version
                gt_lower = ground_truth.lower()
                ids_lower = tokenizer.encode(gt_lower, add_special_tokens=False)
                if len(ids_lower) > 0:
                    target_token_ids.append(ids_lower[0])

                # Check capitalized version (first letter uppercase)
                gt_cap = ground_truth.capitalize()
                ids_cap = tokenizer.encode(gt_cap, add_special_tokens=False)
                if len(ids_cap) > 0:
                    target_token_ids.append(ids_cap[0])

                # Remove duplicates
                target_token_ids = list(set(target_token_ids))

                if len(target_token_ids) > 0:
                    # Check if ANY of the target tokens are in topk
                    # topk_ids: (num_layers, k)
                    # We want to check against multiple targets.

                    # Create a mask for hits
                    hits = torch.zeros(topk_ids.shape[0], device="cpu")

                    for tid in target_token_ids:
                        # (num_layers,) bool
                        current_hits = (topk_ids == tid).any(dim=-1).float().cpu()
                        hits = torch.maximum(hits, current_hits)

                    layer_hits += hits

    if dataset_type == "multiple_choice":
        accuracies_list = []
        if len(all_ground_truths) > 0:
            for l_idx in range(num_layers):
                score = accuracy_score(all_ground_truths, all_predictions[l_idx])
                accuracies_list.append(score)
            return np.array(accuracies_list)
        else:
            print(
                "  ⚠️ Warning: No valid ground truths collected for multiple choice evaluation."
            )
            return np.zeros(num_layers)
    else:
        accuracies = layer_hits / num_samples
        return accuracies.numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Logit Lens Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Results Directory
    results_dir = os.path.expanduser(args.results_dir)
    logitlens_dir = os.path.join(results_dir, "logitlens_evaluation")
    os.makedirs(logitlens_dir, exist_ok=True)

    seed_all(args.seed)

    # Load Model
    print(f"\n🤖 Loading model: {args.model_name_or_path}")
    model_cache_dir = os.path.expanduser(os.getenv("HF_HUB_CACHE"))
    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.model_name_or_path,
        cache_dir=model_cache_dir,
        dtype=MODEL_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="auto",
    )

    model_name_slug = args.model_name_or_path.split("/")[-1]
    datasets_cache_dir = os.path.expanduser(os.getenv("HF_DATASETS_CACHE"))

    # Iterate over Datasets
    for dataset_name, config in DATASETS.items():
        dataset_slug = dataset_name.split("/")[-1].replace("-", "_").lower()
        print(f"\n{'-' * 80}")
        print(f"📚 Dataset: {dataset_name}")
        print(f"{'-' * 80}")

        # Iterate over Subsets
        for subset in config["subsets"]:
            print(f"\n  👉 Subset: {subset}")

            eval_methods = config.get("eval_methods", ["logit_lens"])
            if isinstance(eval_methods, str):
                eval_methods = [eval_methods]

            # Determine which methods need to be run
            methods_to_run = []
            for method in eval_methods:
                if method == "linear_probe" and config.get("type") != "multiple_choice":
                    raise ValueError(
                        f"Linear probing selected for {dataset_name}, but it is not a multiple_choice dataset."
                    )

                # Check if result exists
                result_filename = (
                    f"{model_name_slug}_{dataset_slug}_{subset}_{method}.safetensors"
                )
                result_path = os.path.join(logitlens_dir, result_filename)

                if os.path.exists(result_path):
                    print(
                        f"  ✅ Results already exist at {result_path} for {method}. Skipping."
                    )
                else:
                    methods_to_run.append(method)

            if not methods_to_run:
                continue

            # Load Dataloader
            print("  📥 Loading data...")
            dl_args = {
                "dataset_path_or_name": dataset_name,
                "seed": args.seed,
                "processor": processor,
                "batch_size": args.batch_size,
                "cache_dir": datasets_cache_dir,
                "answer_letters_with_processed_batch": True,
                "split": config.get(
                    "split", "test"
                ),  # Default to test if not specified
                "guide_text": config.get("guide_text", ""),
                "downsample_size": config.get("downsample_size"),
                "remove_yes_no": config.get("remove_yes_no", False),
                "filter_mmbench_4_choices": config.get(
                    "filter_mmbench_4_choices", False
                ),
                "open_ended": config.get("open_ended_prompt", False),
            }

            # Set specific subset flags
            if subset == "texts_qa":
                dl_args["texts_qa"] = True
                dl_args["images_qa"] = False
            elif subset == "images_qa":
                dl_args["texts_qa"] = False
                dl_args["images_qa"] = True
            if "intern" in model.__class__.__name__.lower() or is_cambrian_model(
                model=model
            ):
                dl_args["model"] = model
            if is_cambrian_model(model=model):
                dl_args["num_workers"] = 0

            dataloader = get_dataloader(**dl_args)

            # Extract labels
            # Dataloader yields tuples where labels are always at the end
            all_labels = [label for batch in dataloader for label in batch[-1]]

            # Extract Representations
            print("  🕵️ Extracting representations...")
            residual_stream = residual_stream_tracer(
                model=model,
                processor=processor,
                processed_dataloader=dataloader,
                residual_stream_type=RESIDUAL_STREAM_TYPE,
                tokens_pooling_method=TOKENS_POOLING_METHOD,
                return_deepcopy=True,
            )

            for eval_method in methods_to_run:
                print(f"  🔍 Evaluating with {eval_method}...")
                result_filename = f"{model_name_slug}_{dataset_slug}_{subset}_{eval_method}.safetensors"
                result_path = os.path.join(logitlens_dir, result_filename)

                # Evaluate
                tokenizer = (
                    processor.tokenizer
                    if hasattr(processor, "tokenizer")
                    else processor
                )
                if eval_method == "logit_lens":
                    accuracies = evaluate_logit_lens(
                        model=model,
                        tokenizer=tokenizer,
                        residual_stream=residual_stream,
                        labels=all_labels,
                        dataset_config=config,
                        dataloader=dataloader,
                    )
                elif eval_method == "linear_probe":
                    import numpy as np
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import accuracy_score
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import LabelEncoder

                    print("  🤖 Training Logistic Probes per layer...")

                    # Encode labels (A, B, C, D -> 0, 1, 2, 3)
                    le = LabelEncoder()
                    y = le.fit_transform(all_labels)
                    # print counts of each label
                    unique, counts = np.unique(y, return_counts=True)
                    label_counts = dict(zip(unique, counts))
                    print(f"  Label distribution: {label_counts}")
                    print(f"  Classes: {le.classes_}")

                    num_layers = residual_stream.shape[0]
                    accuracies_list = []

                    # Ensure data is on CPU and numpy
                    res_stream_np = residual_stream.float().cpu().numpy()

                    for layer_idx in range(num_layers):
                        X = res_stream_np[layer_idx]

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=args.seed, stratify=y
                        )

                        # Train Classifier
                        clf = LogisticRegression(max_iter=5000, random_state=args.seed)
                        clf.fit(X_train, y_train)

                        # Evaluate
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        accuracies_list.append(acc)

                        print(f"  Layer {layer_idx}: Test Accuracy = {acc:.4f}")

                    accuracies = np.array(accuracies_list)
                    print(accuracies)

                # Save
                save_file({"accuracies": accuracies}, result_path)
                print(f"  💾 Saved results to {result_path}")

            # Cleanup
            del residual_stream
            gc.collect()
            torch.cuda.empty_cache()

    print("\n✅ All evaluations complete.")


if __name__ == "__main__":
    main()
