from __future__ import annotations

import importlib.util
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import torch


class CambrianBridgeError(RuntimeError):
    """Raised when Cambrian bridge loading is unavailable or fails."""


def _suppress_logging() -> int:
    """Suppress all logging and warnings. Returns previous logging level to restore."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    try:
        from transformers.utils import logging as tf_logging

        tf_logging.set_verbosity_error()
        tf_logging.disable_progress_bar()
    except Exception:
        pass
    return previous


def _ensure_cambrian_importable() -> None:
    if importlib.util.find_spec("cambrian") is not None:
        return

    root = Path(__file__).resolve().parents[1] / "third_party" / "cambrian"
    if not (root / "cambrian").exists():
        raise CambrianBridgeError(
            f"Cambrian source not found at {root}. "
            "Clone it into third_party/cambrian."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if importlib.util.find_spec("cambrian") is None:
        raise CambrianBridgeError(
            f"Cambrian package not importable after adding {root} to sys.path."
        )


def _patch_transformers_cache_compat() -> None:
    try:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = (  # type: ignore[attr-defined]
                lambda self, *_args, **_kwargs: self.get_seq_length(0)
            )
    except ImportError:
        return


def load_cambrian_pretrained(
    model_name_or_path: str,
    cache_dir: str | None = None,
    device_map: str = "auto",
    attn_implementation: str = "flash_attention_2",
    skip_model: bool = False,
    skip_processor: bool = False,
) -> Any:
    if skip_model and skip_processor:
        raise ValueError("Cannot skip both model and processor.")

    _patch_transformers_cache_compat()

    previous_log_level = _suppress_logging()
    try:
        _ensure_cambrian_importable()
        from cambrian.model.builder import load_pretrained_model

        model_name = model_name_or_path.rstrip("/").split("/")[-1]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_flash_attn = attn_implementation == "flash_attention_2" and device == "cuda"

        try:
            tokenizer, model, image_processor, _ = load_pretrained_model(
                model_path=model_name_or_path,
                model_base=None,
                model_name=model_name,
                load_8bit=False,
                load_4bit=False,
                device_map=device_map,
                device=device,
                use_flash_attn=use_flash_attn,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise CambrianBridgeError(
                f"Failed to load Cambrian model '{model_name_or_path}': {exc}"
            ) from exc
    finally:
        logging.disable(previous_log_level)

    if image_processor is not None and not hasattr(tokenizer, "image_processor"):
        tokenizer.image_processor = image_processor

    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    if skip_model:
        return tokenizer
    if skip_processor:
        return model.eval()

    return model.eval(), tokenizer
