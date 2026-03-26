import os
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from rich import print

from utils import load_hf_model_and_processor_or_tokenizer
from utils.cambrian_utils import apply_chat_template, generate_cambrian_batch

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


MODEL_NAME = "lbasile/cambrian-8b-pretrained"
# MODEL_NAME = "nyu-visionx/cambrian-8b"


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def main() -> None:
    print(f"\n[bold]Loading model:[/bold] {MODEL_NAME}")
    model, tokenizer = load_hf_model_and_processor_or_tokenizer(MODEL_NAME)

    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000000139.jpg",
    ]
    images = [load_image_from_url(url) for url in urls]

    print("\n[bold]Single-image QA smoke[/bold]")
    single_questions = ["Describe the scene in detail."]
    prompts = apply_chat_template(model, single_questions, has_image=True)
    responses = generate_cambrian_batch(
        model=model,
        tokenizer=tokenizer,
        questions=single_questions,
        images=[images[0]],
        max_new_tokens=80,
        temperature=0.2,
    )
    print(f"[cyan]Prompt:[/cyan] {prompts[0]}")
    print(f"[green]Response:[/green] {responses[0]}")

    print("\n[bold]Two-image robustness smoke[/bold]")
    questions = [
        "What are the animals doing?",
        "What is on the image?",
    ]
    prompts = apply_chat_template(model, questions, has_image=True)
    responses = generate_cambrian_batch(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        images=images,
        max_new_tokens=80,
    )
    for idx, (prompt, response) in enumerate(zip(prompts, responses), start=1):
        print("-" * 60)
        print(f"[cyan]Sample {idx} prompt:[/cyan] {prompt}")
        print(f"[cyan]Sample {idx} response:[/cyan] {response}")


if __name__ == "__main__":
    main()
