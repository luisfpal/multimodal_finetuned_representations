import os
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import torch
from PIL import Image
from rich import print

from utils import load_hf_model_and_processor_or_tokenizer
from utils.constants import SYSTEM_ROLE


HOME_DIR = Path(os.path.expanduser("~"))
HF_HOME = HOME_DIR / "scratch" / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
HF_DATASETS_CACHE = HF_HOME / "datasets"

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)

print(f"[blue]HF_HOME:            {HF_HOME}[/blue]")
print(f"[blue]HF_HUB_CACHE:       {HF_HUB_CACHE}[/blue]")
print(f"[blue]HF_DATASETS_CACHE:  {HF_DATASETS_CACHE}[/blue]")


MODEL_NAME = "lbasile/llava_7b"
# MODEL_NAME = "l11p/llava-v1.5-7b-pretrained-lbasile"


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def create_conversation(question: str) -> list[dict[str, Any]]:
    return [
        SYSTEM_ROLE,
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]


def render_prompt(processor: Any, conversation: list[dict[str, Any]]) -> str:
    return processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch_responses(
    model: Any,
    processor: Any,
    prompts: list[str],
    images: list[Image.Image],
    max_new_tokens: int = 120,
) -> list[str]:
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    prompt_token_len = inputs["input_ids"].shape[1]
    responses: list[str] = []
    for sample_idx in range(generated.shape[0]):
        continuation_ids = generated[sample_idx][prompt_token_len:]
        responses.append(
            processor.tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True,
            ).strip()
        )
    return responses


def main() -> None:
    print(f"\n[bold]Loading model:[/bold] {MODEL_NAME}")
    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=MODEL_NAME,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000000139.jpg",
    ]
    questions = [
        "What are the animals doing?",
        "What is on the image?",
    ]
    images = [load_image_from_url(url) for url in urls]

    conversations = [create_conversation(question) for question in questions]
    prompts = [render_prompt(processor, conversation) for conversation in conversations]

    print("\n[bold]LLaVA batch smoke[/bold]")
    responses = generate_batch_responses(
        model=model,
        processor=processor,
        prompts=prompts,
        images=images,
        max_new_tokens=120,
    )

    for idx, (prompt, response) in enumerate(zip(prompts, responses), start=1):
        print("-" * 70)
        print(f"[cyan]Sample {idx} prompt:[/cyan] {prompt}")
        print(f"[cyan]Sample {idx} response:[/cyan] {response}")


if __name__ == "__main__":
    main()
