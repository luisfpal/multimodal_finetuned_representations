"""Cambrian overview script — understand prompt, tokens, image flow, and hidden states."""

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import torch
from PIL import Image
from rich import print

from utils import load_hf_model_and_processor_or_tokenizer
from utils.cambrian_utils import (
    generate_cambrian_batch,
    prepare_cambrian_batch_inputs,
)
from utils.context_pooling_utils import get_image_token_indices, resolve_image_token_ids


HOME_DIR = Path(os.path.expanduser("~"))
HF_HOME = HOME_DIR / "scratch" / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
HF_DATASETS_CACHE = HF_HOME / "datasets"

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)


MODEL_NAME = "nyu-visionx/cambrian-8b"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
QUESTIONS = [
    "What is happening in this image?",
    "Describe the scene in detail, including key objects and their spatial relations.",
]
GEN_MAX_NEW_TOKENS = 64
GEN_TEMPERATURE = 0.2


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def print_model_overview(model: Any, tokenizer: Any) -> None:
    print("\n[bold cyan]═══ CAMBRIAN MODEL OVERVIEW ═══[/bold cyan]")
    print(f"[green]Model class:[/green] {type(model).__name__}")
    print(f"[green]Tokenizer class:[/green] {type(tokenizer).__name__}")

    cfg = model.config
    interesting_fields = [
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "image_token_index",
        "image_token_len",
        "image_position",
        "mm_use_im_start_end",
        "start_of_vision_sampler_layers",
        "stride_of_vision_sampler_layers",
    ]
    for field in interesting_fields:
        if hasattr(cfg, field):
            print(f"[blue]{field}:[/blue] {getattr(cfg, field)}")

    if hasattr(model, "hf_device_map"):
        used_devices = sorted({str(v) for v in model.hf_device_map.values()})
        print(f"[blue]hf_device_map devices:[/blue] {used_devices}")
    else:
        print(f"[blue]model.device:[/blue] {model.device}")


def _safe_token_repr(tokenizer: Any, token_id: int) -> str:
    if token_id < 0:
        return f"<special:{token_id}>"
    try:
        token = tokenizer.convert_ids_to_tokens(token_id)
    except (OverflowError, ValueError):
        return f"<unmapped:{token_id}>"
    if token is None:
        return f"<none:{token_id}>"
    return str(token)


def display_tokens(
    input_ids: torch.Tensor,
    image_indices: torch.Tensor,
    tokenizer: Any,
) -> None:
    """Show token regions and detailed token IDs in a single pass."""
    if len(image_indices) == 0:
        print("[yellow]No image tokens found.[/yellow]")
        return

    first_img = int(image_indices[0].item())
    last_img = int(image_indices[-1].item())
    n_img = len(image_indices)

    # Region summary
    if first_img > 0:
        print(f"[green]TEXT (before image):[/green] [0-{first_img - 1}] ({first_img} tokens)")
    print(f"[red]IMAGE:[/red] [{first_img}-{last_img}] ({n_img} tokens)")
    if last_img < len(input_ids) - 1:
        n_after = len(input_ids) - last_img - 1
        print(f"[green]TEXT (after image):[/green] [{last_img + 1}-{len(input_ids) - 1}] ({n_after} tokens)")

    # Detailed tokens
    def _show(start: int, end: int, max_show: int) -> None:
        for i in range(start, min(start + max_show, end)):
            tid = int(input_ids[i].item())
            print(f"  [{i:3d}] ID={tid:6d} → {_safe_token_repr(tokenizer, tid)}")
        remaining = end - start - max_show
        if remaining > 0:
            print(f"  ... ({remaining} more)")

    print("\n[green]═══ TEXT (before image) ═══[/green]")
    _show(0, first_img, 12)
    print("\n[red]═══ IMAGE TOKENS ═══[/red]")
    _show(first_img, last_img + 1, 4)
    if n_img > 4:
        print(f"  ... (repeated {n_img} times total)")
    print("\n[green]═══ TEXT (after image) ═══[/green]")
    _show(last_img + 1, len(input_ids), 24)


def show_prompt_and_tokens(model: Any, tokenizer: Any, image: Image.Image) -> dict[str, Any]:
    print("\n[bold cyan]═══ PROMPT + TOKEN FLOW ═══[/bold cyan]")

    prepared = prepare_cambrian_batch_inputs(
        model=model,
        tokenizer=tokenizer,
        questions=QUESTIONS,
        images=[image] * len(QUESTIONS),
    )
    tokenized = prepared["tokenized"]
    prompts = prepared["prompts"]
    image_tensors = prepared["image_tensors"]
    image_sizes = prepared["image_sizes"]

    print(f"[green]Conversation mode:[/green] {prepared['conv_mode']}")
    print(f"[blue]input_ids shape:[/blue] {tuple(tokenized['input_ids'].shape)}")
    print(f"[blue]attention_mask shape:[/blue] {tuple(tokenized['attention_mask'].shape)}")
    print("\n[cyan]Attention mask:[/cyan]")
    print(tokenized["attention_mask"])

    image_token_ids = resolve_image_token_ids(model, tokenizer)
    print(f"\n[green]Resolved image token IDs:[/green] {image_token_ids}")

    for batch_idx, (prompt, question) in enumerate(zip(prompts, QUESTIONS), start=1):
        input_ids = tokenized["input_ids"][batch_idx - 1]
        attention_mask = tokenized["attention_mask"][batch_idx - 1]
        image_indices = get_image_token_indices(input_ids, image_token_ids)

        print(f"\n[bold]Batch {batch_idx} Question:[/bold] {question}")
        print(
            f"[green]Batch {batch_idx} prompt length (attention sum):[/green] "
            f"{int(attention_mask.sum().item())}"
        )
        print(f"\n[yellow]Batch {batch_idx} rendered prompt:[/yellow]")
        print(prompt)

        head_ids = input_ids[:20].tolist()
        print(f"\n[cyan]Batch {batch_idx} - First 20 token IDs:[/cyan]")
        print(head_ids)

        print(f"\n[green]Batch {batch_idx} num image tokens found:[/green] {len(image_indices)}")
        if len(image_indices) > 0:
            print(
                f"[green]Batch {batch_idx} image token range:[/green] "
                f"{int(image_indices[0].item())} -> {int(image_indices[-1].item())}"
            )

        print(f"\n[bold]Batch {batch_idx} Tokens:[/bold]")
        display_tokens(input_ids, image_indices, tokenizer)

    if isinstance(image_tensors, list):
        for idx, tensor in enumerate(image_tensors):
            print(f"[blue]image_tensors[{idx}] shape:[/blue] {tuple(tensor.shape)}")
    else:
        print(f"[blue]image_tensors shape:[/blue] {tuple(image_tensors.shape)}")
    print(f"[blue]image_sizes:[/blue] {image_sizes}")

    return prepared


def show_hidden_states(model: Any, prepared: dict[str, Any]) -> None:
    print("\n[bold cyan]═══ HIDDEN STATES CHECK ═══[/bold cyan]")
    tokenized = prepared["tokenized"]
    image_tensors = prepared["image_tensors"]
    image_sizes = prepared["image_sizes"]

    batch_size = tokenized["input_ids"].shape[0]
    for batch_idx in range(batch_size):
        sample_input_ids = tokenized["input_ids"][batch_idx : batch_idx + 1]
        sample_attention_mask = tokenized["attention_mask"][batch_idx : batch_idx + 1]
        if isinstance(image_tensors, list):
            sample_images = [t[batch_idx : batch_idx + 1] for t in image_tensors]
        else:
            sample_images = image_tensors[batch_idx : batch_idx + 1]
        sample_image_sizes = [image_sizes[batch_idx]]

        with torch.inference_mode():
            outputs = model(
                input_ids=sample_input_ids,
                attention_mask=sample_attention_mask,
                images=sample_images,
                image_sizes=sample_image_sizes,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        print(f"[green]Batch {batch_idx + 1} hidden states tensors:[/green] {len(hidden_states)}")
        print(
            f"[green]Batch {batch_idx + 1} first hidden state shape:[/green] "
            f"{tuple(hidden_states[0].shape)}"
        )
        print(
            f"[green]Batch {batch_idx + 1} last hidden state shape:[/green] "
            f"{tuple(hidden_states[-1].shape)}"
        )


def run_generation_demo(model: Any, tokenizer: Any, image: Image.Image) -> None:
    print("\n[bold cyan]═══ GENERATION DEMO ═══[/bold cyan]")
    responses = generate_cambrian_batch(
        model=model,
        tokenizer=tokenizer,
        questions=QUESTIONS,
        images=[image] * len(QUESTIONS),
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        temperature=GEN_TEMPERATURE,
    )
    for idx, response in enumerate(responses, start=1):
        print(f"[green]Response {idx}:[/green] {response!r}")


def main() -> None:
    print(f"[blue]HF_HOME:[/blue] {HF_HOME}")
    print(f"[blue]HF_HUB_CACHE:[/blue] {HF_HUB_CACHE}")
    print(f"[blue]HF_DATASETS_CACHE:[/blue] {HF_DATASETS_CACHE}")

    print(f"\n[bold]Loading Cambrian model:[/bold] {MODEL_NAME}")
    model, tokenizer = load_hf_model_and_processor_or_tokenizer(MODEL_NAME)
    image = load_image_from_url(IMAGE_URL)

    print_model_overview(model, tokenizer)
    prepared = show_prompt_and_tokens(model, tokenizer, image)
    show_hidden_states(model, prepared)
    run_generation_demo(model, tokenizer, image)


if __name__ == "__main__":
    main()
