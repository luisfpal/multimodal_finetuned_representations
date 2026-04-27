from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


class CambrianBridgeError(RuntimeError):
    """Raised when Cambrian bridge loading is unavailable or fails."""


@dataclass(frozen=True)
class CambrianVisionEncoderSpec:
    alias: str
    original_index: int
    model_name: str
    token_len: int


CAMBRIAN_VISION_ENCODER_SPECS: tuple[CambrianVisionEncoderSpec, ...] = (
    CambrianVisionEncoderSpec(
        alias="siglip",
        original_index=0,
        model_name="siglip/CLIP-ViT-SO400M-14-384",
        token_len=576,
    ),
    CambrianVisionEncoderSpec(
        alias="clip",
        original_index=1,
        model_name="openai/clip-vit-large-patch14-336",
        token_len=576,
    ),
    CambrianVisionEncoderSpec(
        alias="dinov2",
        original_index=2,
        model_name="facebook/dinov2-giant-res378",
        token_len=576,
    ),
    CambrianVisionEncoderSpec(
        alias="convnext",
        original_index=3,
        model_name="clip-convnext-XXL-multi-stage",
        token_len=9216,
    ),
)

_CAMBRIAN_ENCODER_BY_ALIAS = {
    spec.alias: spec for spec in CAMBRIAN_VISION_ENCODER_SPECS
}
_CAMBRIAN_ENCODER_BY_NAME = {
    spec.model_name.lower(): spec for spec in CAMBRIAN_VISION_ENCODER_SPECS
}


def normalize_cambrian_vision_encoder_selection(
    encoders: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Normalize user Cambrian vision encoder names to canonical aliases."""
    if encoders is None:
        return None

    normalized: list[str] = []
    for encoder in encoders:
        key = encoder.strip().lower()
        if not key:
            continue

        spec = _CAMBRIAN_ENCODER_BY_ALIAS.get(key)
        if spec is None:
            spec = _CAMBRIAN_ENCODER_BY_NAME.get(key)
        if spec is None:
            valid = ", ".join(spec.alias for spec in CAMBRIAN_VISION_ENCODER_SPECS)
            raise ValueError(
                f"Unknown Cambrian vision encoder '{encoder}'. "
                f"Valid aliases: {valid}."
            )
        if spec.alias in normalized:
            raise ValueError(
                f"Duplicate Cambrian vision encoder '{spec.alias}' in selection."
            )
        normalized.append(spec.alias)

    return tuple(normalized) if normalized else None


def build_cambrian_vision_variant_name(
    base_model_name: str,
    encoders: Sequence[str] | None,
) -> str:
    normalized = normalize_cambrian_vision_encoder_selection(encoders)
    if normalized is None:
        return base_model_name
    return f"{base_model_name}__vision-{'+'.join(normalized)}"


def cambrian_vision_selection_metadata(
    encoders: Sequence[str] | None,
) -> dict[str, str]:
    normalized = normalize_cambrian_vision_encoder_selection(encoders)
    if normalized is None:
        return {}
    return {
        "cambrian_vision_encoders": "+".join(normalized),
        "cambrian_vision_encoder_mode": "selected",
    }


def _selected_cambrian_vision_specs(
    encoders: Sequence[str] | None,
) -> tuple[CambrianVisionEncoderSpec, ...] | None:
    normalized = normalize_cambrian_vision_encoder_selection(encoders)
    if normalized is None:
        return None
    return tuple(_CAMBRIAN_ENCODER_BY_ALIAS[alias] for alias in normalized)


def _build_cambrian_branch_key_mapping(
    selected_specs: Sequence[CambrianVisionEncoderSpec],
) -> dict[str, str]:
    selected_old_indices = {spec.original_index for spec in selected_specs}
    key_mapping: dict[str, str] = {}

    for original_index in range(len(CAMBRIAN_VISION_ENCODER_SPECS)):
        if original_index in selected_old_indices:
            continue
        key_mapping[f"\\bmm_projector_aux_{original_index}\\b"] = (
            f"unused_mm_projector_aux_{original_index}"
        )
        key_mapping[f"\\bk_proj_{original_index}\\b"] = (
            f"unused_k_proj_{original_index}"
        )
        key_mapping[f"\\bv_proj_{original_index}\\b"] = (
            f"unused_v_proj_{original_index}"
        )
        key_mapping[f"\\bpos_embed_{original_index}\\b"] = (
            f"unused_pos_embed_{original_index}"
        )
        key_mapping[f"\\baggregate_{original_index}\\b"] = (
            f"unused_aggregate_{original_index}"
        )

    for new_index, spec in enumerate(selected_specs):
        old_index = spec.original_index
        if old_index == new_index:
            continue
        key_mapping[f"\\bmm_projector_aux_{old_index}\\b"] = (
            f"mm_projector_aux_{new_index}"
        )
        key_mapping[f"\\bk_proj_{old_index}\\b"] = f"k_proj_{new_index}"
        key_mapping[f"\\bv_proj_{old_index}\\b"] = f"v_proj_{new_index}"
        key_mapping[f"\\bpos_embed_{old_index}\\b"] = f"pos_embed_{new_index}"
        key_mapping[f"\\baggregate_{old_index}\\b"] = f"aggregate_{new_index}"

    return key_mapping


def _filter_cambrian_config_for_vision_encoders(
    model_name_or_path: str,
    cache_dir: str | None,
    selected_specs: Sequence[CambrianVisionEncoderSpec],
) -> tuple[Any, dict[str, str]]:
    from transformers import AutoConfig

    try:
        config_source: str | Path = _resolve_checkpoint_dir(model_name_or_path, cache_dir)
    except CambrianBridgeError:
        config_source = model_name_or_path
    config = AutoConfig.from_pretrained(config_source, cache_dir=cache_dir)
    projector_type = getattr(config, "mm_projector_type", None)
    if projector_type != "sva":
        raise CambrianBridgeError(
            "Cambrian vision encoder selection is supported only for SVA "
            f"checkpoints. Found mm_projector_type={projector_type!r}."
        )

    original_names = list(getattr(config, "mm_vision_tower_aux_list", []))
    original_token_lens = list(getattr(config, "mm_vision_tower_aux_token_len_list", []))
    expected_names = [spec.model_name for spec in CAMBRIAN_VISION_ENCODER_SPECS]
    expected_token_lens = [spec.token_len for spec in CAMBRIAN_VISION_ENCODER_SPECS]
    if original_names != expected_names or original_token_lens != expected_token_lens:
        raise CambrianBridgeError(
            "Cambrian vision encoder selection expected the official 4-encoder "
            "8B configuration, but the checkpoint config differs. "
            f"Found names={original_names}, token_lens={original_token_lens}."
        )

    config.mm_vision_tower_aux_list = [spec.model_name for spec in selected_specs]
    config.mm_vision_tower_aux_token_len_list = [
        spec.token_len for spec in selected_specs
    ]

    return config, _build_cambrian_branch_key_mapping(selected_specs)


def _resolve_checkpoint_dir(model_name_or_path: str, cache_dir: str | None) -> Path:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return path

    try:
        return Path(
            snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        )
    except Exception as exc:
        raise CambrianBridgeError(
            "Could not resolve a local Cambrian checkpoint snapshot for selected "
            f"vision encoder weight verification: {model_name_or_path}"
        ) from exc


def _remap_selected_branch_key(
    key: str,
    selected_specs: Sequence[CambrianVisionEncoderSpec],
) -> str | None:
    remapped_key = key
    matched = False
    for new_index, spec in enumerate(selected_specs):
        old_index = spec.original_index
        replacements = (
            (rf"\bmm_projector_aux_{old_index}\b", f"mm_projector_aux_{new_index}"),
            (rf"\bk_proj_{old_index}\b", f"k_proj_{new_index}"),
            (rf"\bv_proj_{old_index}\b", f"v_proj_{new_index}"),
            (rf"\bpos_embed_{old_index}\b", f"pos_embed_{new_index}"),
            (rf"\baggregate_{old_index}\b", f"aggregate_{new_index}"),
        )
        for pattern, replacement in replacements:
            remapped_key, count = re.subn(pattern, replacement, remapped_key)
            if count:
                matched = True
                break
        if matched:
            return remapped_key
    return None


def _load_selected_cambrian_connector_weights(
    model: Any,
    model_name_or_path: str,
    cache_dir: str | None,
    selected_specs: Sequence[CambrianVisionEncoderSpec],
) -> None:
    checkpoint_dir = _resolve_checkpoint_dir(model_name_or_path, cache_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise CambrianBridgeError(
            "Selected Cambrian vision encoder loading requires a safetensors "
            f"checkpoint index at {index_path}."
        )

    weight_map = json.loads(index_path.read_text())["weight_map"]
    model_keys = set(model.state_dict().keys())
    remapped_by_shard: dict[str, list[tuple[str, str]]] = {}
    expected_targets: set[str] = set()

    for source_key, shard_name in weight_map.items():
        target_key = _remap_selected_branch_key(source_key, selected_specs)
        if target_key is None:
            continue
        if target_key not in model_keys:
            raise CambrianBridgeError(
                "Selected Cambrian connector remapping produced a key that is "
                f"not present in the filtered model: {source_key} -> {target_key}"
            )
        remapped_by_shard.setdefault(shard_name, []).append((source_key, target_key))
        expected_targets.add(target_key)

    if not expected_targets:
        aliases = ", ".join(spec.alias for spec in selected_specs)
        raise CambrianBridgeError(
            "No Cambrian connector tensors were found for selected vision "
            f"encoders: {aliases}."
        )

    required_targets: set[str] = set()
    for key in model_keys:
        for new_index in range(len(selected_specs)):
            if any(
                re.search(pattern, key)
                for pattern in (
                    rf"\bmm_projector_aux_{new_index}\b",
                    rf"\bk_proj_{new_index}\b",
                    rf"\bv_proj_{new_index}\b",
                    rf"\bpos_embed_{new_index}\b",
                    rf"\baggregate_{new_index}\b",
                )
            ):
                required_targets.add(key)
                break
    missing_targets = sorted(required_targets - expected_targets)
    if missing_targets:
        raise CambrianBridgeError(
            "Selected Cambrian connector remapping did not find source tensors "
            f"for filtered model keys: {missing_targets[:20]}"
        )

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name, key_pairs in remapped_by_shard.items():
        with safe_open(checkpoint_dir / shard_name, framework="pt", device="cpu") as f:
            for source_key, target_key in key_pairs:
                state_dict[target_key] = f.get_tensor(source_key)

    unexpected = model.load_state_dict(state_dict, strict=False).unexpected_keys
    if unexpected:
        raise CambrianBridgeError(
            "Unexpected keys while loading selected Cambrian connector weights: "
            f"{unexpected}"
        )


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
    cambrian_vision_encoders: Sequence[str] | None = None,
) -> Any:
    if skip_model and skip_processor:
        raise ValueError("Cannot skip both model and processor.")

    selected_specs = _selected_cambrian_vision_specs(cambrian_vision_encoders)
    _patch_transformers_cache_compat()

    previous_log_level = _suppress_logging()
    try:
        _ensure_cambrian_importable()
        from cambrian.model.builder import load_pretrained_model

        model_name = model_name_or_path.rstrip("/").split("/")[-1]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_flash_attn = attn_implementation == "flash_attention_2" and device == "cuda"

        try:
            load_kwargs: dict[str, Any] = {}
            if selected_specs is not None:
                filtered_config, key_mapping = (
                    _filter_cambrian_config_for_vision_encoders(
                        model_name_or_path=model_name_or_path,
                        cache_dir=cache_dir,
                        selected_specs=selected_specs,
                    )
                )
                load_kwargs["config"] = filtered_config
                load_kwargs["key_mapping"] = key_mapping

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
                **load_kwargs,
            )
            if selected_specs is not None:
                _load_selected_cambrian_connector_weights(
                    model=model,
                    model_name_or_path=model_name_or_path,
                    cache_dir=cache_dir,
                    selected_specs=selected_specs,
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
