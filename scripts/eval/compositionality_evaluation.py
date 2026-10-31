"""
Compositionality evaluation for generative VLMs on ARO, BiVLC, and SugarCrepe benchmarks.

Evaluates VLMs using constrained MCQ logits: compares logit scores for answer letter
tokens (A, B, ...) at the last input position — no generation, no parsing.

Benchmarks:
  - ARO (Attribution, Relation, Order): 50k+ test cases for compositional failures
    Subtasks: VG-Relation, VG-Attribution, COCO-Order, Flickr30k-Order
  - BiVLC: Bidirectional compositionality (I2T direction; T2I requires multi-image)
  - SugarCrepe: 7-category compositional benchmark (Add/Replace/Swap perturbations)

Supports: LLaVA 1.5/v1.6, LLaVA-OneVision, InternVL2, InternVL3.5, Cambrian

Setup:
  ARO source: third_party/vision-language-models-are-bows/ (checked into repo)
  ARO data:   --aro-data-root (scratch, must contain JSON annotations + images).
              Run: bash shell/data/download_aro_images.sh to download images.
  BiVLC:      Auto-downloads from HuggingFace (imirandam/BiVLC).
  SugarCrepe: --sugarcrepe-data-root (scratch, must contain data/*.json + val2017/).
              Run: bash shell/data/download_sugarcrepe.sh to download.

Usage:
    python scripts/eval/compositionality_evaluation.py \\
        --model-name-or-path lbasile/llava_7b \\
        --output-dir evals/llava_7b/compositionality \\
        --benchmark bivlc \\
        --limit 100
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_project_root_env = os.environ.get("PROJECT_ROOT")
if _project_root_env:
    PROJECT_ROOT = Path(_project_root_env).expanduser().resolve()
else:
    _script_path = Path(__file__).resolve()
    PROJECT_ROOT = (
        _script_path.parents[2]
        if len(_script_path.parents) > 2
        else Path.cwd().resolve()
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nltk
import torch
from datasets import load_dataset
from PIL import Image
from rich import print
from torch.utils.data import Dataset
from tqdm import tqdm

# ARO Order subtasks use nltk.word_tokenize (via TextShuffler) which needs punkt_tab.
# Auto-download once; subsequent calls are no-ops.
nltk.download("punkt_tab", quiet=True)
from utils import seed_all
from utils.cambrian_utils import (
    is_cambrian_model,
    apply_chat_template as apply_cambrian_chat_template,
    prepare_cambrian_batch_inputs,
    get_cambrian_device,
)
from utils.constants import SYSTEM_ROLE
from utils.dataset import _processor_has_default_system
from utils.internvl_utils import (
    apply_chat_template as apply_internvl_chat_template,
    get_combined_embeddings,
    get_pixel_values,
)
from utils.model_utils import (
    is_mllama_model_name_or_path,
    load_hf_model_and_processor_or_tokenizer,
    resolve_attn_implementation_for_model,
)
from utils.patch_llava import patch as patch_llava
from utils.patch_internvl import patch as patch_internvl
from utils.patch_onevision import patch as patch_onevision
from utils.patch_cambrian import patch as patch_cambrian

MODEL_BACKEND_REGISTRY = [
    ("internvl3", "internvl3.5"),
    ("internvl", "internvl2"),
    ("onevision", "llava_onevision1_5"),
    ("cambrian", "cambrian"),
    ("llava", "llava_hf"),
    ("pixtral", "llava_hf"),
]


def infer_backend(model_name: str) -> str:
    name_lower = model_name.lower()
    for pattern, backend in MODEL_BACKEND_REGISTRY:
        if pattern in name_lower:
            return backend
    raise ValueError(
        f"Cannot infer backend for '{model_name}'. "
        f"Known patterns: {[p for p, _ in MODEL_BACKEND_REGISTRY]}"
    )


def _num_language_layers(model, backend: str) -> int:
    if backend == "cambrian":
        return model.config.num_hidden_layers
    return model.language_model.config.num_hidden_layers


def apply_no_image_patch(model, backend: str) -> None:
    num_layers = _num_language_layers(model, backend)
    layers = range(num_layers)
    if backend == "llava_onevision1_5":
        patch_onevision(model, layers_to_skip=layers)
    elif backend == "llava_hf":
        patch_llava(model, layers_to_skip=layers)
    elif backend in ("internvl2", "internvl3.5"):
        patch_internvl(model, layers_to_skip=layers)
    elif backend == "cambrian":
        patch_cambrian(model, layers_to_skip=layers)
    else:
        raise ValueError(f"--no-image not supported for backend: {backend}")

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = 42
DEFAULT_DATASETS_CACHE = os.environ.get(
    "HF_DATASETS_CACHE",
    os.path.expanduser("~/scratch/huggingface/datasets"),
)
DEFAULT_ARO_DATA_ROOT = str(
    Path.home() / "scratch" / "datasets" / "vision-language-models-are-bows"
)
# ARO source code (dataset_zoo/) lives here — checked into the repo under third_party/
ARO_SOURCE_ROOT = str(
    PROJECT_ROOT / "third_party" / "vision-language-models-are-bows"
)

ARO_SUBTASKS = ("flickr30k_order", "coco_order", "vg_relation", "vg_attribution")
BIVLC_DATASET_ID = "imirandam/BiVLC"

# SugarCrepe — 7 categories across Add / Replace / Swap perturbation types.
# Images: COCO val2017. JSONs from github.com/RAIVNLab/sugar-crepe.
SUGARCREPE_CATEGORIES = (
    "add_obj",
    "add_att",
    "replace_obj",
    "replace_att",
    "replace_rel",
    "swap_obj",
    "swap_att",
)
# Paper-standard display labels (SugarCrepe NeurIPS 2023, Table 3 column order)
SUGARCREPE_LABELS: dict[str, str] = {
    "add_obj": "Add-Obj",
    "add_att": "Add-Att",
    "replace_obj": "Rep-Obj",
    "replace_att": "Rep-Att",
    "replace_rel": "Rep-Rel",
    "swap_obj": "Swap-Obj",
    "swap_att": "Swap-Att",
}
# Type groupings for aggregate reporting
SUGARCREPE_GROUPS: dict[str, tuple[str, ...]] = {
    "Add": ("add_obj", "add_att"),
    "Replace": ("replace_obj", "replace_att", "replace_rel"),
    "Swap": ("swap_obj", "swap_att"),
}
DEFAULT_SUGARCREPE_ROOT = str(Path.home() / "scratch" / "datasets" / "sugar-crepe")

# Default sample limits per subtask (full dataset sizes in comments).
# CLI --limit overrides these.
TASK_LIMITS = {
    "flickr30k_order": 5000,  # 4885 samples
    "coco_order": 5000,       # 25010 samples
    "vg_relation": 5000,      # 23937 samples
    "vg_attribution": 5000,   # 28748 samples
    "bivlc": 5000,            # 2933 samples
    # SugarCrepe per-category limits (comment out to use full dataset)
    # "add_obj": 2062,          # 2062 samples
    # "add_att": 692,           # 692 samples
    # "replace_obj": 1652,      # 1652 samples
    # "replace_att": 788,       # 788 samples
    # "replace_rel": 1406,      # 1406 samples
    # "swap_obj": 245,          # 245 samples
    # "swap_att": 666,          # 666 samples
}

# MCQ prompt templates
MCQ_BINARY_PROMPT = (
    "Which caption better describes this image?\n"
    "A) {caption_a}\n"
    "B) {caption_b}\n"
    "Answer with the option's letter from the given choices directly."
)
MCQ_5WAY_PROMPT = (
    "Which caption best describes this image?\n"
    "A) {caption_a}\n"
    "B) {caption_b}\n"
    "C) {caption_c}\n"
    "D) {caption_d}\n"
    "E) {caption_e}\n"
    "Answer with the option's letter from the given choices directly."
)


def build_mcq_prompt(captions: list[str]) -> str:
    """Build MCQ prompt from a list of captions."""
    num_choices = len(captions)
    if num_choices == 2:
        return MCQ_BINARY_PROMPT.format(caption_a=captions[0], caption_b=captions[1])
    if num_choices == 5:
        return MCQ_5WAY_PROMPT.format(
            caption_a=captions[0],
            caption_b=captions[1],
            caption_c=captions[2],
            caption_d=captions[3],
            caption_e=captions[4],
        )
    # Fallback: build dynamically
    lines = [f"{chr(ord('A') + i)}) {cap}" for i, cap in enumerate(captions)]
    return (
        "Which caption best describes this image?\n"
        + "\n".join(lines)
        + "\nAnswer with the letter only."
    )


def _shuffle_captions(
    captions: list[str],
    correct_idx: int,
    idx: int,
) -> tuple[list[str], int, list[int]]:
    """Deterministically shuffle captions to avoid position bias.

    Returns (shuffled_captions, new_correct_idx, shuffle_order).
    shuffle_order[i] = original index that ended up at position i.
    """
    rng = random.Random(RANDOM_SEED + idx)
    order = list(range(len(captions)))
    rng.shuffle(order)
    shuffled = [captions[i] for i in order]
    new_correct = order.index(correct_idx)
    return shuffled, new_correct, order


# =============================================================================
# MODEL TYPE DETECTION
# =============================================================================


def detect_model_type(model_name_or_path: str) -> str:
    """Detect model family from model name/path string."""
    if is_mllama_model_name_or_path(model_name_or_path):
        return "mllama"
    name_lower = model_name_or_path.lower()
    if "cambrian" in name_lower:
        return "cambrian"
    if "internvl3" in name_lower:
        return "internvl3"
    if "internvl" in name_lower:
        return "internvl2"
    # LLaVA family (including OneVision, Qwen2-VL variants)
    return "llava"


def is_internvl_model(model: torch.nn.Module) -> bool:
    """Check if model is InternVL by class name."""
    return "intern" in model.__class__.__name__.lower()


def is_mllama_model(model: torch.nn.Module) -> bool:
    """Check if model is Mllama / Llama Vision by class name."""
    return "mllama" in model.__class__.__name__.lower()


# =============================================================================
# MCQ PROMPT BUILDERS — shared by constrained logits evaluation
# =============================================================================


def _get_internvl_img_context_token_id(tokenizer: Any) -> int:
    """Get the img_context_token_id for InternVL models."""
    from utils.internvl_utils import IMG_CONTEXT_TOKEN

    return tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)


def _build_llava_prompt(
    processor: Any,
    image: Optional[Image.Image],
    question: str,
) -> tuple[str, dict]:
    """Build LLaVA prompt and processor inputs for MCQ. Returns (prompt_text, inputs_dict)."""
    include_system = not _processor_has_default_system(processor)
    user_content: list = []
    if image is not None:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": question})

    conv: list = [{"role": "user", "content": user_content}]
    if include_system:
        conv.insert(0, SYSTEM_ROLE)
    prompt_text = processor.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True
    )

    proc_kwargs = {"text": prompt_text, "return_tensors": "pt", "padding": False}
    if image is not None:
        proc_kwargs["images"] = image

    inputs = processor(**proc_kwargs)
    inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    return prompt_text, inputs


def _build_mllama_prompt(
    processor: Any,
    image: Optional[Image.Image],
    question: str,
) -> tuple[str, dict]:
    """Build Llama Vision prompt and processor inputs for MCQ."""
    user_content: list = []
    if image is not None:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": question})

    prompt_text = processor.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    proc_kwargs = {
        "text": [prompt_text],
        "return_tensors": "pt",
        "padding": False,
        "add_special_tokens": False,
    }
    if image is not None:
        proc_kwargs["images"] = [[image]]

    inputs = processor(**proc_kwargs)
    inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    return prompt_text, inputs


def _clean_internvl_prompt_for_saving(query: str) -> str:
    """Remove InternVL image placeholder tokens for human-readable saved prompt."""
    from utils.internvl_utils import IMG_CONTEXT_TOKEN

    cleaned = query.replace(IMG_CONTEXT_TOKEN, "")
    cleaned = cleaned.replace("</img>", "").replace("<img>", "<image>")
    return cleaned


def _build_internvl_inputs(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
) -> tuple[str, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Build InternVL inputs for MCQ. Returns (cleaned_prompt, input_ids, attention_mask, inputs_embeds_or_None)."""
    device = model.device
    pixel_values = None
    num_patches_list = None

    if image is not None:
        pv, num_patches_list = get_pixel_values(image)
        pixel_values = pv.to(device)

    queries = apply_internvl_chat_template(
        model=model,
        text=[question],
        num_patches_list=num_patches_list,
    )
    query = queries[0]
    cleaned_prompt = _clean_internvl_prompt_for_saving(query)

    input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    img_ctx_id = _get_internvl_img_context_token_id(tokenizer)
    inputs_embeds = None
    if pixel_values is not None:
        inputs_embeds = get_combined_embeddings(
            model, input_ids, pixel_values, img_context_token_id=img_ctx_id
        )

    return cleaned_prompt, input_ids, attention_mask, inputs_embeds


# =============================================================================
# MCQ EVALUATION — CONSTRAINED LOGITS
# =============================================================================


def _get_answer_token_ids(
    tokenizer: Any,
    num_choices: int,
) -> list[int]:
    """Get token IDs for answer letters A, B, C, ... up to num_choices."""
    ids = []
    for i in range(num_choices):
        letter = chr(ord("A") + i)
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        # Single letters are single tokens in all supported tokenizers
        ids.append(encoded[0])
    return ids


def _pick_answer_from_logits(
    logits: torch.Tensor,
    answer_token_ids: list[int],
) -> tuple[int, str]:
    """Pick the answer letter with highest logit among answer tokens.

    Args:
        logits: logits at the last input position, shape (vocab_size,)
        answer_token_ids: token IDs for answer letters [A, B, ...]

    Returns:
        (predicted_index, predicted_letter) — 0-indexed answer and letter string.
    """
    answer_logits = logits[answer_token_ids]
    predicted = int(answer_logits.argmax().item())
    letter = chr(ord("A") + predicted)
    return predicted, letter


def _get_mcq_logits_llava(
    model: torch.nn.Module,
    processor: Any,
    image: Optional[Image.Image],
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Constrained logits MCQ for LLaVA. Returns (predicted_idx, predicted_letter, prompt)."""
    prompt_text, inputs = _build_llava_prompt(processor, image, question)
    model_dtype = next(model.parameters()).dtype
    moved = {}
    for k, v in inputs.items():
        v = v.to(model.device)
        if v.is_floating_point():
            v = v.to(model_dtype)
        moved[k] = v
    inputs = moved

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits at last input position
    last_logits = outputs.logits[0, -1, :]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, prompt_text


def _get_mcq_logits_mllama(
    model: torch.nn.Module,
    processor: Any,
    image: Optional[Image.Image],
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Constrained logits MCQ for Llama Vision / Mllama."""
    prompt_text, inputs = _build_mllama_prompt(processor, image, question)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    last_logits = outputs.logits[0, -1, :]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, prompt_text


def _get_mcq_logits_internvl(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Constrained logits MCQ for InternVL. Returns (predicted_idx, predicted_letter, prompt)."""
    cleaned_prompt, input_ids, attention_mask, inputs_embeds = _build_internvl_inputs(
        model, tokenizer, image, question
    )

    from utils.internvl_utils import IMG_CONTEXT_TOKEN
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    if (input_ids == model.img_context_token_id).any():
        image_limits = []
        image_attention_mask = torch.ones_like(input_ids, dtype=torch.int)
        for i in range(input_ids.shape[0]):
            indices = torch.where(input_ids[i] == model.img_context_token_id)[0]
            if indices.numel() > 0:
                start = indices[0].item() - 1
                end = indices[-1].item() + 1
                image_limits.append((start, end))
                image_attention_mask[i, start : end + 1] = 0
            else:
                image_limits.append(None)

        model.language_model.model.image_limits = image_limits
        model.language_model.model.image_attention_mask = image_attention_mask

    with torch.no_grad():
        if inputs_embeds is not None:
            outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        else:
            outputs = model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    last_logits = outputs.logits[0, -1, :]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, cleaned_prompt


def _get_mcq_logits_cambrian(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Constrained logits MCQ for Cambrian. Returns (predicted_idx, predicted_letter, prompt)."""
    sample_images = [image] if image is not None else None
    prepared = prepare_cambrian_batch_inputs(
        model=model,
        tokenizer=tokenizer,
        questions=[question],
        images=sample_images,
    )
    prompt_text = prepared["prompts"][0]

    input_ids = prepared["tokenized"]["input_ids"]
    attention_mask = prepared["tokenized"]["attention_mask"]
    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if prepared["image_tensors"] is not None:
        forward_kwargs["images"] = prepared["image_tensors"]
        forward_kwargs["image_sizes"] = prepared["image_sizes"]
    
    with torch.inference_mode():
        outputs = model(**forward_kwargs)

    last_logits = outputs.logits[0, -1, :]
    predicted_idx, letter = _pick_answer_from_logits(last_logits, answer_token_ids)
    return predicted_idx, letter, prompt_text


def get_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    image: Optional[Image.Image],
    question: str,
    answer_token_ids: list[int],
) -> tuple[int, str, str]:
    """Constrained logits MCQ, dispatching to model-specific implementation.

    Returns (predicted_index, predicted_letter, prompt_text).
    """
    if is_cambrian_model(model):
        return _get_mcq_logits_cambrian(
            model, processor_or_tokenizer, image, question, answer_token_ids
        )
    if is_internvl_model(model):
        return _get_mcq_logits_internvl(
            model, processor_or_tokenizer, image, question, answer_token_ids
        )
    if is_mllama_model(model):
        return _get_mcq_logits_mllama(
            model, processor_or_tokenizer, image, question, answer_token_ids
        )
    return _get_mcq_logits_llava(
        model, processor_or_tokenizer, image, question, answer_token_ids
    )


# =============================================================================
# LIMIT RESOLUTION
# =============================================================================


def _resolve_limit(subtask: str, cli_limit: Optional[int]) -> Optional[int]:
    """Resolve sample limit: CLI --limit overrides per-subtask defaults."""
    if cli_limit is not None:
        return cli_limit
    return TASK_LIMITS.get(subtask)


# =============================================================================
# ARO DATASETS
# =============================================================================


def _ensure_aro_on_sys_path() -> None:
    """Add ARO source (third_party/) to sys.path so dataset_zoo can be imported."""
    if not os.path.isdir(os.path.join(ARO_SOURCE_ROOT, "dataset_zoo")):
        raise ImportError(
            f"ARO source not found at {ARO_SOURCE_ROOT}. "
            f"Expected third_party/vision-language-models-are-bows/ to be present in the repo."
        )
    if ARO_SOURCE_ROOT not in sys.path:
        sys.path.insert(0, ARO_SOURCE_ROOT)


def _ensure_nltk_punkt_tab() -> None:
    """Ensure ARO Order text shuffling can use nltk.word_tokenize."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


class AROVGDataset(Dataset):
    """ARO VG-Relation or VG-Attribution dataset.

    Source code loaded from third_party/; data (JSON annotations + images) from aro_data_root.
    """

    def __init__(
        self,
        subtask: str,
        aro_data_root: str,
        limit: Optional[int] = None,
    ):
        if subtask not in ("vg_relation", "vg_attribution"):
            raise ValueError(f"Invalid ARO VG subtask: {subtask}")

        self.subtask = subtask
        _ensure_aro_on_sys_path()
        data_path = os.path.expanduser(aro_data_root)

        from dataset_zoo import VG_Relation, VG_Attribution

        DatasetClass = VG_Relation if subtask == "vg_relation" else VG_Attribution
        self.dataset = DatasetClass(
            image_preprocess=None, download=True, root_dir=data_path
        )
        n = len(self.dataset)
        if limit and limit < n:
            rng = random.Random(RANDOM_SEED)
            self.indices = sorted(rng.sample(range(n), limit))
        else:
            self.indices = list(range(n))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[self.indices[idx]]
        image = item["image_options"][0]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        return {
            "image": image,
            "caption_options": item["caption_options"],
            "correct_index": 1,  # true caption is at index 1
        }


class AROOrderDataset(Dataset):
    """ARO COCO-Order or Flickr30k-Order dataset.

    Source code loaded from third_party/; images loaded from aro_data_root.
    """

    def __init__(
        self,
        subtask: str,
        aro_data_root: str,
        limit: Optional[int] = None,
    ):
        if subtask not in ("coco_order", "flickr30k_order"):
            raise ValueError(f"Invalid ARO Order subtask: {subtask}")

        self.subtask = subtask
        _ensure_aro_on_sys_path()
        _ensure_nltk_punkt_tab()
        data_path = os.path.expanduser(aro_data_root)

        from dataset_zoo import COCO_Order, Flickr30k_Order

        DatasetClass = COCO_Order if subtask == "coco_order" else Flickr30k_Order
        # Order datasets need split="test"; Flickr30k_Order requires it as positional arg
        self.dataset = DatasetClass(
            image_preprocess=None, split="test", download=True, root_dir=data_path
        )
        n = len(self.dataset)
        if limit and limit < n:
            rng = random.Random(RANDOM_SEED)
            self.indices = sorted(rng.sample(range(n), limit))
        else:
            self.indices = list(range(n))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[self.indices[idx]]
        image = item["image_options"][0]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        return {
            "image": image,
            "caption_options": item["caption_options"],
            "correct_index": 0,  # original caption is at index 0
        }


def load_aro_dataset(
    subtask: str,
    aro_data_root: str,
    limit: Optional[int] = None,
) -> Dataset:
    """Load an ARO subtask dataset."""
    if subtask in ("vg_relation", "vg_attribution"):
        return AROVGDataset(subtask, aro_data_root, limit)
    return AROOrderDataset(subtask, aro_data_root, limit)


_ARO_ORDER_IMAGE_DIRS: dict[str, tuple[str, str]] = {
    "coco_order": ("val2014", "COCO val2014"),
    "flickr30k_order": ("flickr30k-images", "Flickr30k"),
}


def validate_aro_image_dirs(subtasks: list[str], aro_data_root: str) -> None:
    """Fail-fast: verify image directories exist before loading model."""
    aro_path = os.path.expanduser(aro_data_root)
    missing = []
    for st in subtasks:
        if st in _ARO_ORDER_IMAGE_DIRS:
            subdir, name = _ARO_ORDER_IMAGE_DIRS[st]
            if not os.path.isdir(os.path.join(aro_path, subdir)):
                missing.append((st, name, os.path.join(aro_path, subdir)))
    if missing:
        lines = ["Missing image directories for ARO order subtasks:"]
        for st, name, path in missing:
            lines.append(f"  - {st}: {name} images not found at {path}")
        lines.append("")
        lines.append("Download them with:")
        lines.append(
            f"  python scripts/data/download_aro_images.py --aro-data-root {aro_path}"
        )
        raise FileNotFoundError("\n".join(lines))


# =============================================================================
# BiVLC DATASET
# =============================================================================


def load_bivlc_dataset(
    cache_dir: str,
    limit: Optional[int] = None,
) -> list:
    """Load BiVLC dataset from HuggingFace. Returns list of dicts."""
    dataset = load_dataset(BIVLC_DATASET_ID, split="test", cache_dir=cache_dir)
    if limit and limit < len(dataset):
        rng = random.Random(RANDOM_SEED)
        indices = sorted(rng.sample(range(len(dataset)), limit))
        dataset = dataset.select(indices)
    return dataset


# =============================================================================
# SUGARCREPE DATASET
# =============================================================================


def load_sugarcrepe_data(
    data_root: str,
    category: str,
    limit: Optional[int] = None,
) -> list[dict]:
    """Load one SugarCrepe category JSON. Returns list of dicts with filename/caption/negative_caption."""
    json_path = os.path.join(data_root, "data", f"{category}.json")
    with open(json_path) as f:
        raw = json.load(f)
    samples = list(raw.values())
    if limit and limit < len(samples):
        rng = random.Random(RANDOM_SEED)
        samples = [samples[i] for i in sorted(rng.sample(range(len(samples)), limit))]
    return samples


def validate_sugarcrepe_data(data_root: str) -> None:
    """Fail fast if SugarCrepe JSON files or COCO val2017 images directory are missing."""
    data_dir = os.path.join(data_root, "data")
    img_dir = os.path.join(data_root, "val2017")
    missing = [
        c
        for c in SUGARCREPE_CATEGORIES
        if not os.path.isfile(os.path.join(data_dir, f"{c}.json"))
    ]
    if missing:
        raise FileNotFoundError(
            f"SugarCrepe JSON files missing for: {missing}\n"
            f"  Expected in: {data_dir}\n"
            f"  Run: bash shell/data/download_sugarcrepe.sh"
        )
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"COCO val2017 image directory not found: {img_dir}\n"
            f"  Run: bash shell/data/download_sugarcrepe.sh"
        )



# =============================================================================
# BENCHMARK EVALUATION — ARO + BiVLC + SugarCrepe (MCQ LOGITS)
# =============================================================================


def evaluate_aro_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    subtask: str,
    aro_data_root: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a single ARO subtask using constrained logits MCQ.

    Compares logits of answer letter tokens (A, B, ...) at the last input
    position. The letter with the highest logit wins. No generation or parsing.
    """
    dataset = load_aro_dataset(subtask, aro_data_root, limit)

    # Determine tokenizer for answer token IDs
    tokenizer = (
        processor_or_tokenizer.tokenizer
        if hasattr(processor_or_tokenizer, "tokenizer")
        else processor_or_tokenizer
    )
    # Pre-compute answer token IDs (max possible choices across ARO subtasks)
    max_choices = 5
    all_answer_ids = _get_answer_token_ids(tokenizer, max_choices)

    correct = 0
    total = 0
    per_sample: list = []

    n_samples = len(dataset)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(
        total=n_samples, desc=f"ARO {subtask} (mcq_logits)", unit="sample"
    )
    update_count = 0

    skipped = 0
    for idx in range(n_samples):
        try:
            item = dataset[idx]
        except (FileNotFoundError, OSError):
            skipped += 1
            update_count += 1
            if update_count >= update_every or idx == n_samples - 1:
                progress_bar.update(update_count)
                update_count = 0
            continue

        image = item["image"]
        captions = item["caption_options"]
        original_correct_idx = item["correct_index"]
        num_choices = len(captions)
        answer_token_ids = all_answer_ids[:num_choices]

        # Shuffle to avoid position bias
        shuffled, correct_idx, shuffle_order = _shuffle_captions(
            captions, original_correct_idx, idx
        )
        prompt = build_mcq_prompt(shuffled)

        try:
            predicted_idx, predicted_token, prompt_text = get_mcq_logits(
                model, processor_or_tokenizer, image, prompt, answer_token_ids
            )
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            predicted_idx = None
            predicted_token = ""
            prompt_text = prompt

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append(
            {
                "idx": idx,
                "prompt": prompt_text,
                "predicted_token": predicted_token,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
                "shuffle_order": shuffle_order,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()
    if skipped:
        print(
            f"[yellow]  ⚠ Skipped {skipped}/{n_samples} samples (missing images)[/yellow]"
        )

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
        "num_skipped": skipped,
        "per_sample": per_sample,
    }


def evaluate_bivlc_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    cache_dir: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate BiVLC I2T using constrained logits MCQ.

    T2I is skipped (requires multi-image prompting). Use log-likelihood for T2I.
    """
    dataset = load_bivlc_dataset(cache_dir, limit)

    tokenizer = (
        processor_or_tokenizer.tokenizer
        if hasattr(processor_or_tokenizer, "tokenizer")
        else processor_or_tokenizer
    )
    answer_token_ids = _get_answer_token_ids(tokenizer, 2)

    i2t_correct = 0
    total = 0
    per_type: Dict[str, Dict[str, int]] = {}
    per_sample: list = []

    n_samples = len(dataset)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(total=n_samples, desc="BiVLC I2T (mcq_logits)", unit="sample")
    update_count = 0

    for idx in range(n_samples):
        item = dataset[idx]
        image = item["image"]
        caption = item["caption"]
        neg_caption = item["negative_caption"]
        comp_type = item.get("type", "unknown")
        comp_subtype = item.get("subtype", "unknown")

        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Randomize order to avoid position bias
        options = [caption, neg_caption]
        options, correct_idx, shuffle_order = _shuffle_captions(options, 0, idx)

        prompt = build_mcq_prompt(options)

        try:
            predicted_idx, predicted_token, prompt_text = get_mcq_logits(
                model, processor_or_tokenizer, image, prompt, answer_token_ids
            )
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            predicted_idx = None
            predicted_token = ""
            prompt_text = prompt

        i2t_ok = predicted_idx == correct_idx
        if i2t_ok:
            i2t_correct += 1
        total += 1

        type_key = f"{comp_type}/{comp_subtype}"
        if type_key not in per_type:
            per_type[type_key] = {"i2t": 0, "total": 0}
        per_type[type_key]["total"] += 1
        if i2t_ok:
            per_type[type_key]["i2t"] += 1

        per_sample.append(
            {
                "idx": idx,
                "prompt": prompt_text,
                "predicted_token": predicted_token,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": i2t_ok,
                "type": comp_type,
                "subtype": comp_subtype,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()

    per_type_acc = {}
    for key, counts in per_type.items():
        n = counts["total"]
        per_type_acc[key] = {
            "i2t": counts["i2t"] / n if n > 0 else 0.0,
            "total": n,
        }

    return {
        "i2t_accuracy": i2t_correct / total if total > 0 else 0.0,
        "num_total": total,
        "per_type": per_type_acc,
        "per_sample": per_sample,
        "note": "T2I not evaluated (requires multi-image support).",
    }


# =============================================================================
# BENCHMARK EVALUATION — SUGARCREPE
# =============================================================================


def evaluate_sugarcrepe_mcq_logits(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    data_root: str,
    category: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one SugarCrepe category using constrained logits MCQ.

    Compares logits of answer letter tokens (A, B) at the last input position.
    No generation or parsing — always produces a valid answer.
    """
    samples = load_sugarcrepe_data(data_root, category, limit)
    img_dir = os.path.join(data_root, "val2017")

    tokenizer = (
        processor_or_tokenizer.tokenizer
        if hasattr(processor_or_tokenizer, "tokenizer")
        else processor_or_tokenizer
    )
    answer_token_ids = _get_answer_token_ids(tokenizer, 2)

    correct = 0
    total = 0
    per_sample: list = []

    n_samples = len(samples)
    update_every = max(1, int(0.1 * n_samples))
    progress_bar = tqdm(
        total=n_samples, desc=f"SugarCrepe {category} (mcq_logits)", unit="sample"
    )
    update_count = 0

    for idx, sample in enumerate(samples):
        image = Image.open(os.path.join(img_dir, sample["filename"])).convert("RGB")
        caption = sample["caption"]
        neg_caption = sample["negative_caption"]

        options, correct_idx, shuffle_order = _shuffle_captions(
            [caption, neg_caption], 0, idx
        )
        prompt = build_mcq_prompt(options)

        try:
            predicted_idx, predicted_token, prompt_text = get_mcq_logits(
                model, processor_or_tokenizer, image, prompt, answer_token_ids
            )
        except Exception as e:
            print(f"[yellow]⚠ Error at sample {idx}: {e}[/yellow]")
            predicted_idx = None
            predicted_token = ""
            prompt_text = prompt

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        per_sample.append(
            {
                "idx": idx,
                "filename": sample["filename"],
                "prompt": prompt_text,
                "predicted_token": predicted_token,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
                "shuffle_order": shuffle_order,
            }
        )

        update_count += 1
        if update_count >= update_every or idx == n_samples - 1:
            progress_bar.update(update_count)
            update_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "num_correct": correct,
        "num_total": total,
        "per_sample": per_sample,
    }


# =============================================================================
# RESULT SAVING
# =============================================================================


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time as Xh Ym Zs."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m > 0 else f"{seconds:.1f}s"


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    model_identifier: str,
    benchmark: str,
    eval_method: str,
    elapsed_sec: float,
) -> None:
    """Save evaluation results to JSONL (metrics) and JSON (per-sample)."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract per-sample data and build summary
    per_sample = results.pop("per_sample", [])
    summary = {
        "experiment_name": f"single/{model_identifier}",
        "eval_method": eval_method,
        **results,
        "elapsed_time": format_elapsed_time(elapsed_sec),
    }

    # Append metrics to JSONL
    jsonl_path = os.path.join(output_dir, f"{benchmark}_results.jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[green]  Metrics → {jsonl_path}[/green]")

    # Save per-sample responses (include benchmark in filename to avoid overwrites)
    suffix = f"_{eval_method}"
    json_path = os.path.join(
        output_dir, f"{benchmark}_{model_identifier}{suffix}_responses.json"
    )
    with open(json_path, "w") as f:
        json.dump(per_sample, f, indent=2, default=str)
    print(f"[green]  Samples → {json_path}[/green]")


# =============================================================================
# ORCHESTRATION
# =============================================================================


def run_aro_evaluation(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    args: argparse.Namespace,
    model_identifier: str,
) -> None:
    seed_all(42)
    """Run full ARO evaluation (mcq_logits method)."""
    print("[bold magenta]=== ARO Compositionality Benchmark ===[/bold magenta]")

    all_accs: dict[str, float] = {}

    for subtask in args.aro_subtasks:
        resolved_limit = _resolve_limit(subtask, args.limit)
        print(f"\n[cyan]▶ {subtask}[/cyan]")
        if resolved_limit:
            print(f"[yellow]  Sample limit: {resolved_limit}[/yellow]")
        output_dir = os.path.join(args.output_dir, "aro")
        start = time.time()

        try:
            results = evaluate_aro_mcq_logits(
                model,
                processor_or_tokenizer,
                subtask,
                args.aro_data_root,
                limit=resolved_limit,
            )
            elapsed = time.time() - start
            print(
                f"[green]  ✓ {subtask}: "
                f"{results['accuracy']:.4f} ({format_elapsed_time(elapsed)})[/green]"
            )
            save_results(results, output_dir, model_identifier, f"aro_{subtask}", "mcq_logits", elapsed)
            all_accs[subtask] = results["accuracy"]
        except Exception as e:
            print(f"[red]  ✗ {subtask}: {e}[/red]")

    if all_accs:
        mean_acc = sum(all_accs.values()) / len(all_accs)
        print(f"\n[bold cyan]── ARO mean: {mean_acc:.4f} ──[/bold cyan]")


def run_bivlc_evaluation(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    args: argparse.Namespace,
    model_identifier: str,
) -> None:
    seed_all(42)
    """Run full BiVLC evaluation (mcq_logits method, I2T direction)."""
    print("[bold magenta]=== BiVLC Compositionality Benchmark ===[/bold magenta]")

    output_dir = os.path.join(args.output_dir, "bivlc")
    resolved_limit = _resolve_limit("bivlc", args.limit)
    if resolved_limit:
        print(f"[yellow]  Sample limit: {resolved_limit}[/yellow]")

    print("\n[cyan]▶ BiVLC (mcq_logits)[/cyan]")
    start = time.time()
    try:
        results = evaluate_bivlc_mcq_logits(
            model,
            processor_or_tokenizer,
            cache_dir=args.datasets_cache_dir,
            limit=resolved_limit,
        )
        elapsed = time.time() - start
        print(
            f"[green]  ✓ I2T: {results['i2t_accuracy']:.4f} "
            f"({format_elapsed_time(elapsed)})[/green]"
        )
        save_results(results, output_dir, model_identifier, "bivlc", "mcq_logits", elapsed)
    except Exception as e:
        print(f"[red]  ✗ BiVLC: {e}[/red]")


def run_sugarcrepe_evaluation(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    args: argparse.Namespace,
    model_identifier: str,
) -> None:
    """Run full SugarCrepe evaluation (mcq_logits method).

    Results are saved as:
      - Per-category response JSONs
      - One aggregate JSONL entry with per_category, per_type, and mean accuracy
        in paper-standard column order (Add-Obj/Add-Att/Rep-Obj/Rep-Att/Rep-Rel/Swap-Obj/Swap-Att).
    """

    seed_all(42)
    print("[bold magenta]=== SugarCrepe Compositionality Benchmark ===[/bold magenta]")

    output_dir = os.path.join(args.output_dir, "sugarcrepe")
    os.makedirs(output_dir, exist_ok=True)

    print("\n[cyan]▶ SugarCrepe (mcq_logits)[/cyan]")
    method_start = time.time()

    category_accs: dict[str, float] = {}
    total_samples = 0

    for category in args.sugarcrepe_categories:
        limit = _resolve_limit(category, args.limit)
        start = time.time()
        try:
            results = evaluate_sugarcrepe_mcq_logits(
                model,
                processor_or_tokenizer,
                args.sugarcrepe_data_root,
                category,
                limit,
            )
            elapsed = time.time() - start
            acc = results["accuracy"]
            category_accs[category] = acc
            total_samples += results["num_total"]

            print(
                f"[green]  ✓ {SUGARCREPE_LABELS[category]}: "
                f"{acc * 100:.1f}% "
                f"({results['num_correct']}/{results['num_total']}, "
                f"{format_elapsed_time(elapsed)})[/green]"
            )

            per_sample = results.pop("per_sample", [])
            resp_path = os.path.join(
                output_dir,
                f"sugarcrepe_{category}_{model_identifier}_mcq_logits_responses.json",
            )
            with open(resp_path, "w") as f:
                json.dump(per_sample, f, indent=2, default=str)
            print(f"[green]    Samples → {resp_path}[/green]")

        except Exception as e:
            print(f"[red]  ✗ {category}: {e}[/red]")
            import traceback

            traceback.print_exc()

    if not category_accs:
        return

    # Build paper-standard aggregate (per_category, per_type, mean)
    per_category = {
        SUGARCREPE_LABELS[c]: category_accs[c]
        for c in SUGARCREPE_CATEGORIES
        if c in category_accs
    }
    per_type: dict[str, float] = {}
    for grp, cats in SUGARCREPE_GROUPS.items():
        done = [c for c in cats if c in category_accs]
        if done:
            per_type[grp] = sum(category_accs[c] for c in done) / len(done)
    mean_acc = sum(per_category.values()) / len(per_category)
    total_elapsed = time.time() - method_start

    # Print paper-ready table
    col_order = [SUGARCREPE_LABELS[c] for c in SUGARCREPE_CATEGORIES if c in category_accs]
    header = "  ".join(f"{lbl:>7}" for lbl in col_order)
    values = "  ".join(f"{per_category[lbl] * 100:>7.1f}" for lbl in col_order)
    grp_header = "  ".join(f"{g:>7}" for g in per_type)
    grp_values = "  ".join(f"{v * 100:>7.1f}" for v in per_type.values())
    print(f"\n[bold cyan]  SugarCrepe — paper table[/bold cyan]")
    print(f"  {header}  │  {grp_header}  │    Mean")
    print(f"  {values}  │  {grp_values}  │  {mean_acc * 100:>6.2f}%")

    aggregate = {
        "experiment_name": f"single/{model_identifier}",
        "eval_method": "mcq_logits",
        "per_category": per_category,
        "per_type": per_type,
        "mean": mean_acc,
        "num_total": total_samples,
        "elapsed_time": format_elapsed_time(total_elapsed),
    }
    jsonl_path = os.path.join(output_dir, "sugarcrepe_results.jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(aggregate) + "\n")
    print(f"[green]  Aggregate → {jsonl_path}[/green]")


# =============================================================================
# MAIN
# =============================================================================


def cleanup_memory() -> None:
    """Release GPU memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on ARO, BiVLC, and SugarCrepe compositionality benchmarks",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="HuggingFace model identifier or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["aro", "bivlc", "sugarcrepe", "all"],
        help="Which benchmark to evaluate (default: all)",
    )
    parser.add_argument(
        "--aro-subtasks",
        type=str,
        nargs="+",
        default=list(ARO_SUBTASKS),
        choices=list(ARO_SUBTASKS),
        help="ARO subtasks to evaluate (default: all 4)",
    )
    parser.add_argument(
        "--aro-data-root",
        type=str,
        default=DEFAULT_ARO_DATA_ROOT,
        help="Path to ARO data root in scratch (images + JSON annotations)",
    )
    parser.add_argument(
        "--sugarcrepe-data-root",
        type=str,
        default=DEFAULT_SUGARCREPE_ROOT,
        help="Path to SugarCrepe data dir (must contain data/*.json and val2017/)",
    )
    parser.add_argument(
        "--sugarcrepe-categories",
        type=str,
        nargs="+",
        default=list(SUGARCREPE_CATEGORIES),
        choices=list(SUGARCREPE_CATEGORIES),
        help="SugarCrepe categories to evaluate (default: all 7)",
    )
    parser.add_argument(
        "--datasets-cache-dir",
        type=str,
        default=DEFAULT_DATASETS_CACHE,
        help="Cache directory for datasets (default: $HF_DATASETS_CACHE)",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=os.environ.get("HF_HUB_CACHE"),
        help="Cache directory for model weights (default: $HF_HUB_CACHE)",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        help=(
            "Attention implementation. Use auto to choose sdpa for Mllama "
            "and flash_attention_2 otherwise."
        ),
    )
    parser.add_argument(
        "--eval-method",
        type=str,
        default="mcq",
        choices=["mcq", "mcq_logits"],
        help="Compatibility flag; compositionality currently uses MCQ logits only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per subtask/category (for debugging)",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Skip vision layers for text-only (image-blind) evaluation",
    )
    args = parser.parse_args()
    try:
        args.attn_implementation = resolve_attn_implementation_for_model(
            args.model_name_or_path,
            args.attn_implementation,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Print configuration
    print(
        "[bold cyan]╔══════════════════════════════════════════════════════════╗[/bold cyan]"
    )
    print(
        "[bold cyan]║         COMPOSITIONALITY EVALUATION                      ║[/bold cyan]"
    )
    print(
        "[bold cyan]╚══════════════════════════════════════════════════════════╝[/bold cyan]"
    )
    print(f"[cyan]Model:     {args.model_name_or_path}[/cyan]")
    print(f"[cyan]Benchmark: {args.benchmark}[/cyan]")
    print(f"[cyan]Method:    mcq_logits[/cyan]")
    print(f"[cyan]Attention: {args.attn_implementation}[/cyan]")
    if args.benchmark in ("sugarcrepe", "all"):
        print(f"[cyan]SugarCrepe: {args.sugarcrepe_data_root}[/cyan]")
    if args.limit:
        print(f"[yellow]Limit:     {args.limit} samples[/yellow]")
    print()

    # Validate data availability before expensive model load
    if args.benchmark in ("aro", "all"):
        validate_aro_image_dirs(args.aro_subtasks, args.aro_data_root)
    if args.benchmark in ("sugarcrepe", "all"):
        validate_sugarcrepe_data(args.sugarcrepe_data_root)

    # Load model
    print("[dim]Loading model...[/dim]")
    model, processor_or_tokenizer = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.model_cache_dir,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    model_identifier = args.model_name_or_path.replace("/", "_")
    print(f"[green]✓ Model loaded: {model.__class__.__name__}[/green]")

    if args.no_image:
        backend = infer_backend(args.model_name_or_path)
        apply_no_image_patch(model, backend)
        print(f"[yellow]✓ No-image patch applied ({backend})[/yellow]")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    overall_start = time.time()

    # Run benchmarks
    # BiVLC is disabled: --benchmark bivlc currently runs nothing.
    # if args.benchmark in ("bivlc", "all"):
    #     try:
    #         run_bivlc_evaluation(model, processor_or_tokenizer, args, model_identifier)
    #     except Exception as e:
    #         print(f"[red]✗ BiVLC evaluation failed: {e}[/red]")
    #         import traceback

    #         traceback.print_exc()

    if args.benchmark in ("aro", "all"):
        try:
            run_aro_evaluation(model, processor_or_tokenizer, args, model_identifier)
        except Exception as e:
            print(f"[red]✗ ARO evaluation failed: {e}[/red]")
            import traceback

            traceback.print_exc()

    if args.benchmark in ("sugarcrepe", "all"):
        try:
            run_sugarcrepe_evaluation(
                model, processor_or_tokenizer, args, model_identifier
            )
        except Exception as e:
            print(f"[red]✗ SugarCrepe evaluation failed: {e}[/red]")
            import traceback

            traceback.print_exc()

    overall_elapsed = time.time() - overall_start
    print()
    print(
        f"[bold green]✓ All evaluations complete ({format_elapsed_time(overall_elapsed)})[/bold green]"
    )
    print(f"[cyan]  Results: {args.output_dir}[/cyan]")

    cleanup_memory()


if __name__ == "__main__":
    main()
