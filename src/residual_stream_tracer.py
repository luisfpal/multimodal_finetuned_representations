import abc
import copy
import gc
import os
import time
from typing import Dict, Optional, Union, Literal, List

import torch
from einops import einsum
from safetensors.torch import save_file, load_file
from rich import print
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature
from typing_extensions import deprecated
from utils import ModelType, ProcessorType, sample_image_and_text_positions
from utils.cambrian_utils import is_cambrian_model
from utils.context_pooling_utils import (
    get_end_user_role_indices,
    get_image_token_indices,
    get_text_context_indices,
    get_tokenizer_from_processor,
    get_user_text_indices,
    resolve_image_token_ids,
)
from utils.internvl_utils import (
    IMG_CONTEXT_TOKEN,
    generate_responses,
    get_combined_embeddings,
)

# Constants for valid parameter values
VALID_RESIDUAL_STREAM_TYPES = [
    "output_layer",
    "post_mlp",
    "heads_projection",
    "post_input_layernorm",
]
VALID_TOKENS_POOLING_METHODS = [
    "mean",
    "last",
    "none",
    "sample",
    "mean_image",
    "last_image",
    "end_image",
    "last10_image",
    "last_img_and_next",
    "mean_text_context",
    "last_caption",
    "end_user_role",
    "last_user_text",
]

# All pooling methods that operate on image-token positions.
# Used to route through the Cambrian expansion path and index computation.
_IMAGE_BASED_POOLING_METHODS: frozenset[str] = frozenset([
    "mean_image",
    "last_image",
    "end_image",
    "last10_image",
    "last_img_and_next",
])

HOOK_BASED_TRACER_DEPRECATION_MSG = (
    "Hook-based residual stream tracers are deprecated and kept only as reference. "
    "They are not maintained with the latest tracing features. "
    "Prefer residual_stream_type='output_layer'."
)


def _resolve_decoder_config(model: ModelType):
    """Resolve decoder config across supported VLM wrappers.

    Supports:
    - LLaVA/InternVL wrappers exposing ``model.language_model.config``
    - Cambrian wrappers exposing ``model.model.config``
    - Fallback to ``model.config`` when available
    """
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "config"):
        return language_model.config

    decoder_model = getattr(model, "model", None)
    if decoder_model is not None and hasattr(decoder_model, "config"):
        return decoder_model.config

    if hasattr(model, "config"):
        return model.config

    raise ValueError(
        "Could not resolve decoder config for residual stream tracing. "
        f"Model class: {model.__class__.__name__}"
    )


class ResidualStreamTracer(abc.ABC):
    """
    Base class for extracting residual streams from huggingface decoder-based multimodal models.

    This class provides common functionality for all residual stream extraction methods including:
    - Common initialization and storage management
    - Token pooling methods
    - Memory management and cleanup
    """

    def __init__(
        self,
        model: ModelType,
        num_samples: int,
        processor: Optional[ProcessorType] = None,
        return_dtype: torch.dtype = torch.bfloat16,
        tokens_pooling_method: Optional[str] = "last",
        embeddings_sampling_args: Optional[dict] = None,
    ):
        """
        Args:
            model (ModelType): huggingface multimodal decoder-based model
            num_samples (int): number of samples to trace, e.g. number of samples in the dataset
            return_dtype (torch.dtype, optional): dtype of the residual stream. Defaults to torch.bfloat16.
            tokens_pooling_method (Optional[str], optional): method to pool the tokens. Defaults to "last".
                This is relatively memory lightweight and fast.
                If None, the residual stream is returned as is but this is memory intensive.
                This is used to compute the prompt entropy and the extraction is very fast but the memory is bounded by
                O(num_samples * num_layers * max_sequence_length * hidden_size * num_bytes).
                !For example, for a model with 32 layers, 4096 hidden size, max sequence length of 678, 2500 samples,
                !and float16 dtype, the memory is bounded by 2500 * 32 * 678 * 4096 * 2 ~ 414 GB.
                Therefore, unless necessary and there is enough memory, using tokens_pooling_method=None is not recommended.
            embeddings_sampling_args (Optional[dict], optional): arguments for sampling embeddings from specific positions.
                If provided, this overrides tokens_pooling_method and samples embeddings from specific image/text positions.
                Expected keys: 'image_token_id', 'image_seq_length', 'pad_token_id', 'text_direction',
                'skip_image_pos', 'skip_text_pos', 'generator'.
        """
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = get_tokenizer_from_processor(processor)
        self.image_token_ids = resolve_image_token_ids(model, processor)
        self.tokens_pooling_method = tokens_pooling_method
        self.embeddings_sampling_args = embeddings_sampling_args
        decoder_config = _resolve_decoder_config(self.model)
        self.num_layers = decoder_config.num_hidden_layers
        self.hidden_size = decoder_config.hidden_size
        self.counter = 0
        self.return_dtype = return_dtype
        self.num_samples = num_samples
        self.num_tokens = 0

        # Validate tokens_pooling_method
        if tokens_pooling_method not in VALID_TOKENS_POOLING_METHODS:
            raise ValueError(
                f"tokens_pooling_method must be one of {VALID_TOKENS_POOLING_METHODS}"
            )

        # Sanity check
        if tokens_pooling_method == "sample" and embeddings_sampling_args is None:
            raise ValueError(
                "embeddings_sampling_args must be provided when tokens_pooling_method is 'sample'"
            )

        self._validate_embeddings_sampling_args()

        print(
            f"🔧 Residual stream extraction with dtype {self.return_dtype}. "
            "Precision depends on dtype. torch.bfloat16 gives high precision but is memory intensive. "
            "Results are equivalent up to dtype precision regardless of batch size."
        )

        self._calculate_memory_usage()
        self._warn_memory_intensive_usage()

        # Initialize storage - to be implemented by subclasses
        self._initialize_storage()

    def _validate_embeddings_sampling_args(self):
        # If embeddings_sampling_args is provided, validate it
        if self.embeddings_sampling_args is not None:
            required_keys = [
                "image_token_id",
                "image_seq_length",
                "pad_token_id",
                "text_direction",
                "skip_image_pos",
                "skip_text_pos",
                "generator",
                "chat_mode",
            ]
            missing_keys = [
                key for key in required_keys if key not in self.embeddings_sampling_args
            ]
            if missing_keys:
                raise ValueError(
                    f"embeddings_sampling_args missing required keys: {missing_keys}"
                )
            if self.embeddings_sampling_args.get(
                "skip_image_pos"
            ) and self.embeddings_sampling_args.get("skip_text_pos"):
                raise ValueError("Cannot skip both image and text positions")
            elif not self.embeddings_sampling_args.get(
                "skip_image_pos"
            ) and not self.embeddings_sampling_args.get("skip_text_pos"):
                raise ValueError("Must skip either image or text positions")

    def _calculate_memory_usage(self):
        """Calculate memory usage per token for the tracer."""
        self.memory_usage_per_token = (
            self.num_layers
            * self.num_samples
            * self.hidden_size
            * self._dtype_size(self.return_dtype)
            / 1024**3
        )

    def _warn_memory_intensive_usage(self):
        """Warn about memory intensive usage when appropriate."""
        if self.tokens_pooling_method == "none":
            print(
                "⚠️ Warning: This tracer is extremely memory intensive. "
                "May require hundreds of GBs of memory. "
                "Consider counting tokens in dataset and multiplying by memory usage per token."
            )

    @abc.abstractmethod
    def _initialize_storage(self):
        """Initialize the storage for residual stream data. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_storage")

    def _dtype_size(self, dtype: torch.dtype) -> int:
        return torch.tensor([], dtype=dtype).element_size()

    def _sample_embeddings_from_positions(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Sample embeddings from specific image/text positions."""
        if not hasattr(self, "current_batch_input_ids"):
            raise ValueError("current_batch_input_ids is required for sample pooling")

        sampling_args = self.embeddings_sampling_args or {}
        positions = sample_image_and_text_positions(
            input_ids=self.current_batch_input_ids,
            **sampling_args,
        )

        # Two positions per batch element (image and text)
        batch_indices = torch.arange(
            hidden_states.shape[0], device=hidden_states.device
        )
        # Take the first position (image) if both are sampled, otherwise use the single position
        return hidden_states[batch_indices, positions]

    def _compute_context_pooling_indices(
        self,
        input_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Compute per-sample indices for context-based pooling modes.

        Supports ``mean_image`` (image token positions) and
        ``mean_text_context`` (user-turn text span).

        Note: ``mean_context`` is a *virtual* pooling method dispatched at
        the experiment-script level (e.g. ``multimodality_similarity.py``)
        by splitting it into ``mean_image`` for image batches and
        ``mean_text_context`` for text batches.  It is intentionally NOT
        handled here — passing it directly will raise a validation error.
        """
        if input_ids is None:
            raise ValueError(
                f"input_ids are required for pooling '{self.tokens_pooling_method}'"
            )

        pooling_indices: list[torch.Tensor] = []
        batch_length = input_ids.shape[0]

        for idx in range(batch_length):
            seq_len = int(sequence_lengths[idx])
            current_input_ids = input_ids[idx, -seq_len:]

            if self.tokens_pooling_method in _IMAGE_BASED_POOLING_METHODS:
                # All image-based methods share the same image-token span; the
                # specific pooling logic (mean, last, last10, +1, etc.) is resolved
                # downstream in _pool_tokens.
                pooling_indices.append(
                    get_image_token_indices(
                        current_input_ids,
                        self.image_token_ids,
                    )
                )
            elif self.tokens_pooling_method in ["mean_text_context", "last_caption"]:
                # Both mean_text_context and last_caption use the same text-context span;
                # mean vs last distinction is resolved downstream in _pool_tokens.
                pooling_indices.append(
                    get_text_context_indices(
                        current_input_ids,
                        tokenizer=self.tokenizer,
                    )
                )
                # keep this for "debugging"
                # decoded_caption_text = self.tokenizer.decode(
                #     current_input_ids[pooling_indices[-1]].tolist(),
                #     skip_special_tokens=False,
                # )
                # print(
                #     f"Decoded text context for sample {self.counter + idx}: {decoded_caption_text!r}"
                # )
            elif self.tokens_pooling_method == "end_user_role":
                pooling_indices.append(
                    get_end_user_role_indices(current_input_ids)
                )
            elif self.tokens_pooling_method == "last_user_text":
                pooling_indices.append(
                    get_user_text_indices(
                        current_input_ids,
                        tokenizer=self.tokenizer,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported pooling method for context indices: "
                    f"'{self.tokens_pooling_method}'"
                )

        return pooling_indices

    def _resolve_batch_pooling_indices(
        self,
        batch_inputs: BatchFeature,
        batch_input_ids: Optional[torch.Tensor],
        sequence_lengths: torch.Tensor,
    ) -> Optional[Union[torch.Tensor, list[torch.Tensor]]]:
        """Resolve pooling indices for a batch, shared by all trace_batch implementations.

        Returns ``None`` when the current pooling method does not require
        per-token indices (``mean``, ``last``, ``none``).
        """
        if self.tokens_pooling_method == "sample":
            source_input_ids = batch_input_ids
            if source_input_ids is None:
                source_input_ids = getattr(batch_inputs, "input_ids", None)
            if source_input_ids is None:
                raise ValueError("input_ids are required for sample pooling")
            sampling_args = self.embeddings_sampling_args or {}
            return sample_image_and_text_positions(
                input_ids=source_input_ids,
                **sampling_args,
            )

        if self.tokens_pooling_method in (
            *_IMAGE_BASED_POOLING_METHODS,
            "mean_text_context",
            "last_caption",
            "end_user_role",
            "last_user_text",
        ):
            source_input_ids = batch_input_ids
            if source_input_ids is None:
                source_input_ids = getattr(batch_inputs, "input_ids", None)
            return self._compute_context_pooling_indices(
                input_ids=source_input_ids,
                sequence_lengths=sequence_lengths,
            )

        return None

    def _pool_tokens(
        self,
        hidden_states: torch.Tensor,
        pooling_indices: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Pool the tokens of the hidden states, one sample at a time.
        The padding tokens are removed before pooling in the respective tracer.

        Args:
            hidden_states: Input tensor to pool
            pooling_indices: Indices to use for pooling (e.g., sampled positions or image token indices).
        Returns:
            Pooled tensor or original tensor if no pooling
        """
        if (
            self.tokens_pooling_method not in ("mean", "last", "none")
            and pooling_indices is None
        ):
            raise ValueError(
                f"pooling_indices must be provided when tokens_pooling_method is '{self.tokens_pooling_method}'"
            )

        ndims = hidden_states.ndim
        if self.tokens_pooling_method == "mean":
            if ndims == 3:  # hidden_states shape -> (layers, max_seq_len, hidden_size)
                return hidden_states.mean(dim=1)  # (layers, hidden_size)
            elif ndims == 2:  # hidden_states shape -> (max_seq_len, hidden_size)
                return hidden_states.mean(dim=0)  # (hidden_size)
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for mean pooling: {ndims}"
                )
        elif self.tokens_pooling_method == "last":
            # Handle different tensor shapes for last token pooling
            if ndims == 2:  # hidden_states shape -> (max_seq_len, hidden_size)
                return hidden_states[-1, :]
            elif (
                ndims == 3
            ):  # hidden_states shape -> (layers, max_seq_len, hidden_size)
                return hidden_states[:, -1, :]
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for last token pooling: {ndims}"
                )
        elif self.tokens_pooling_method == "none":
            return hidden_states
        elif self.tokens_pooling_method == "sample":
            # case 1: (layers, seq_len(idx), hidden_size) or (num_heads, seq_len(idx), hidden_size)
            if ndims == 3:
                return hidden_states[:, pooling_indices, :]
            # case 2: (seq_len(idx), hidden_size)
            elif ndims == 2:
                return hidden_states[pooling_indices, :]
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for sampling: {hidden_states.ndim}"
                )
        elif self.tokens_pooling_method in [
            "mean_image",
            "mean_text_context",
        ]:
            # Guard: empty indices → fail loudly instead of producing NaN
            if len(pooling_indices) == 0:
                raise ValueError(
                    f"Empty pooling indices for '{self.tokens_pooling_method}' — "
                    "cannot compute mean over 0 tokens. Check that the input "
                    "contains the expected token type (image tokens for mean_image, "
                    "user text span for mean_text_context)."
                )
            # case 1: (layers, seq_len(idx), hidden_size) or (num_heads, seq_len(idx), hidden_size)
            # The indices in pooling_indices select the dimension 1 (seq_len)
            if ndims == 3:
                return hidden_states[:, pooling_indices, :].mean(dim=1)
            # case 2: (seq_len(idx), hidden_size)
            elif ndims == 2:
                return hidden_states[pooling_indices, :].mean(dim=0)
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for context pooling: {hidden_states.ndim}"
                )

        elif self.tokens_pooling_method in ["last_caption", "last_image"]:
            # Guard: empty indices → fail loudly
            if len(pooling_indices) == 0:
                raise ValueError(
                    f"Empty pooling indices for '{self.tokens_pooling_method}' — "
                    "cannot select last token from 0 tokens. Check that the input "
                    "contains the expected token type (text span for last_caption, "
                    "image tokens for last_image)."
                )
            # pooling_indices is torch.arange(start, end); pooling_indices[-1]
            # is the index of the last token in the span (caption or image).
            # case 1: (layers, seq_len(idx), hidden_size)
            if ndims == 3:
                return hidden_states[:, pooling_indices[-1], :]
            # case 2: (seq_len(idx), hidden_size)
            elif ndims == 2:
                return hidden_states[pooling_indices[-1], :]
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for {self.tokens_pooling_method} pooling: {hidden_states.ndim}"
                )

        elif self.tokens_pooling_method == "end_image":
            # Single token immediately after the last image token.
            if len(pooling_indices) == 0:
                raise ValueError(
                    "Empty pooling indices for 'end_image' — no image tokens found."
                )
            idx_next = pooling_indices[-1] + 1
            if ndims == 3:
                return hidden_states[:, idx_next, :]
            elif ndims == 2:
                return hidden_states[idx_next, :]
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for end_image pooling: {ndims}"
                )

        elif self.tokens_pooling_method == "last10_image":
            # Mean over the last (up to) 10 image tokens.
            if len(pooling_indices) == 0:
                raise ValueError(
                    "Empty pooling indices for 'last10_image' — no image tokens found."
                )
            last10 = pooling_indices[-10:]
            if ndims == 3:
                return hidden_states[:, last10, :].mean(dim=1)
            elif ndims == 2:
                return hidden_states[last10, :].mean(dim=0)
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for last10_image pooling: {ndims}"
                )

        elif self.tokens_pooling_method == "last_img_and_next":
            # Mean of the last image token and the token immediately after it.
            if len(pooling_indices) == 0:
                raise ValueError(
                    "Empty pooling indices for 'last_img_and_next' — no image tokens found."
                )
            pair = torch.stack(
                [pooling_indices[-1], pooling_indices[-1] + 1]
            )
            if ndims == 3:
                return hidden_states[:, pair, :].mean(dim=1)
            elif ndims == 2:
                return hidden_states[pair, :].mean(dim=0)
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for last_img_and_next pooling: {ndims}"
                )

        elif self.tokens_pooling_method == "end_user_role":
            # Mean over the anchor tokens at the end of the user role.
            # Indices come from get_end_user_role_indices (model-auto-detected).
            if len(pooling_indices) == 0:
                raise ValueError(
                    "Empty pooling indices for 'end_user_role' — anchor not found."
                )
            if ndims == 3:
                return hidden_states[:, pooling_indices, :].mean(dim=1)
            elif ndims == 2:
                return hidden_states[pooling_indices, :].mean(dim=0)
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for end_user_role pooling: {ndims}"
                )

        elif self.tokens_pooling_method == "last_user_text":
            # Last token of {text_context} in a pile-10k 'Continue this text' prompt.
            if len(pooling_indices) == 0:
                raise ValueError(
                    "Empty pooling indices for 'last_user_text' — text_context span not found."
                )
            if ndims == 3:
                return hidden_states[:, pooling_indices[-3], :]
            elif ndims == 2:
                return hidden_states[pooling_indices[-3], :]
            else:
                raise ValueError(
                    f"Unsupported tensor dimensions for last_user_text pooling: {ndims}"
                )

        else:
            raise ValueError(
                f"Invalid tokens pooling method: {self.tokens_pooling_method}"
            )

    @abc.abstractmethod
    def trace_batch(
        self,
        batch_inputs: BatchFeature,
        batch_input_ids: Optional[torch.Tensor] = None,
    ):
        """
        Trace a tokenized/processed batch of inputs. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement trace_batch")

    def get_residual_stream(
        self,
        return_deepcopy: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get the residual stream of the model as a deepcopy of the internal residual stream.
        """
        if self.tokens_pooling_method == "none":
            mean_tokens_per_sample = self.num_tokens / self.num_samples
            print(
                f"\n🔎 Total tokens in dataset: {self.num_tokens}"
                f"\n📊 Mean tokens per sample: {mean_tokens_per_sample:.2f}"
                f"\n💾 Estimated memory usage: {mean_tokens_per_sample * self.memory_usage_per_token:.2f} GB"
            )

        if return_deepcopy:
            return copy.deepcopy(self._residual_stream).to("cpu")
        else:
            return self._residual_stream.to("cpu")

    def clear(self):
        """
        Clear intermediate variables.
        """
        self.num_tokens = 0
        self.counter = 0


class HookBasedResidualStreamTracer(ResidualStreamTracer):
    """
    Base class for residual stream tracers that use forward hooks.

    This intermediate class provides common functionality for hook-based extraction including:
    - Hook management
    - Attention mask handling
    - Batch processing for hook-based methods
    """

    def __init__(
        self,
        model: ModelType,
        num_samples: int,
        processor: Optional[ProcessorType] = None,
        return_dtype: torch.dtype = torch.bfloat16,
        tokens_pooling_method: Optional[str] = None,
        embeddings_sampling_args: Optional[dict] = None,
    ):
        """
        Initialize hook-based tracer with additional hook management attributes.
        """
        # Initialize parent class first
        super().__init__(
            model=model,
            num_samples=num_samples,
            processor=processor,
            return_dtype=return_dtype,
            tokens_pooling_method=tokens_pooling_method,
            embeddings_sampling_args=embeddings_sampling_args,
        )

        # Hook-specific attributes
        self.handles = []
        self.current_batch_start_idx = 0
        self.sequence_lengths = None
        self.current_batch_pooling_indices = None

        # Set up hooks after initialization
        self._trace()

    @abc.abstractmethod
    def _initialize_none_pooling_storage(self, batch_length: int):
        """
        Initialize storage for None pooling method. To be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _initialize_none_pooling_storage"
        )

    @abc.abstractmethod
    def _trace(self):
        """Set up hooks for the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _trace")

    def trace_batch(
        self,
        batch_inputs: BatchFeature,
        batch_input_ids: Optional[torch.Tensor] = None,
    ):
        """
        Trace the batch inputs using hooks.
        """
        with torch.no_grad():
            batch_length = batch_inputs.attention_mask.shape[0]
            self.sequence_lengths = batch_inputs.attention_mask.sum(dim=1)  # (batch,)
            self.num_tokens += self.sequence_lengths.sum().item()
            # Store the start index for this batch
            self.current_batch_start_idx = self.counter

            # Initialize storage for None pooling if needed
            if self.tokens_pooling_method == "none":
                self._initialize_none_pooling_storage(batch_length)

            # Resolve pooling indices for this batch
            self.current_batch_pooling_indices = self._resolve_batch_pooling_indices(
                batch_inputs, batch_input_ids, self.sequence_lengths
            )

            # Increment counter for next batch
            self.counter += batch_length
            if "intern" in self.model.__class__.__name__.lower():
                _ = self.model.language_model(
                    **batch_inputs,
                    use_cache=False,
                )  # Forward pass to trigger hooks
            else:
                _ = self.model(**batch_inputs, use_cache=False)

    def clear(self):
        """Clean up hooks and cached data."""
        # Remove hooks
        for h in self.handles:
            h.remove()
        self.handles.clear()

        # Clean up hook-specific data
        if hasattr(self, "sequence_lengths"):
            del self.sequence_lengths
        self.current_batch_start_idx = 0

        super().clear()


def _get_lm_components(model) -> tuple:
    """Return (lm_head, final_norm) for logit-lens projection across VLM architectures.

    LLaVA/Cambrian: lm_head is on the top-level model; language_model property returns
    the bare backbone (LlamaModel, no lm_head).
    InternVL2: language_model is a full causal LM (has its own lm_head / model.norm).
    Returns (None, None) when architecture is unrecognised — callers skip silently.
    """
    # lm_head: top-level first (LLaVA, Cambrian), fall back to language_model child (InternVL2)
    lm_head = getattr(model, "lm_head", None)
    lang = getattr(model, "language_model", None)
    if lm_head is None and lang is not None:
        lm_head = getattr(lang, "lm_head", None)

    # norm: lang may be bare backbone (LlamaModel, has .norm directly) or full causal LM
    # (.model.norm); fall back through both paths.
    backbone = lang if lang is not None else model
    inner = getattr(backbone, "model", backbone)
    norm = getattr(inner, "norm", None) or getattr(backbone, "norm", None)

    return lm_head, norm


class ResidualStreamOutputLayerTracer(ResidualStreamTracer):
    """
    This class is used to extract the residual stream of a huggingface
    decoder-based multimodal model, based on the LLaVA architecture,
    at the output of a decoder layer.

    The residual stream can be pooled in different ways:
    - mean: average the residual stream over the sequence length
    - last: take the last token of the sequence
    - none: return the residual stream as is

    The residual stream is returned as a tensor of shape (num_layers, num_samples, hidden_size)
    if the tokens pooling method is not None, or as a dictionary of num_samples tensors of shape
    (num_layers, sequence_length(idx), hidden_size) if the tokens pooling method is None.
    """

    def __init__(
        self,
        model: ModelType,
        num_samples: int,
        processor: Optional[ProcessorType] = None,
        return_dtype: torch.dtype = torch.bfloat16,
        tokens_pooling_method: Optional[str] = None,
        embeddings_sampling_args: Optional[dict] = None,
        token_log: Optional[List[str]] = None,
        layer_token_log: Optional[List[List[str]]] = None,
    ):
        self.token_log = token_log
        self.layer_token_log = layer_token_log
        self._lm_head: Optional[torch.nn.Module] = None
        self._lm_norm: Optional[torch.nn.Module] = None
        if layer_token_log is not None:
            self._lm_head, self._lm_norm = _get_lm_components(model)
            if self._lm_head is None:
                print("  [logit-lens] LM head not found — per-layer token log disabled.")
                self.layer_token_log = None
        super().__init__(
            model=model,
            num_samples=num_samples,
            processor=processor,
            return_dtype=return_dtype,
            tokens_pooling_method=tokens_pooling_method,
            embeddings_sampling_args=embeddings_sampling_args,
        )
        print(
            f"\nThe estimated memory usage per token is {self.memory_usage_per_token:.2f} GB"
        )

    def _initialize_storage(self):
        """Initialize storage for output residual stream."""
        if self.tokens_pooling_method in [
            "mean",
            "last",
            "sample",
            "mean_image",
            "last_image",
            "end_image",
            "last10_image",
            "last_img_and_next",
            "mean_text_context",
            "last_caption",
            "end_user_role",
            "last_user_text",
        ]:
            self._residual_stream = torch.zeros(
                self.num_layers,
                self.num_samples,
                self.hidden_size,
                device="cpu",
                dtype=self.return_dtype,
            )
        elif self.tokens_pooling_method == "none":
            self._residual_stream = {
                str(sample_idx): None for sample_idx in range(self.num_samples)
            }
            # each element is a tensor of shape (num_layers, seq_len(idx), hidden_size)

    def trace_batch(
        self,
        batch_inputs: BatchFeature,
        batch_input_ids: Optional[torch.Tensor] = None,
    ):
        """
        Trace a tokenized/processed batch of inputs of samples along all the layers of the model.
        """
        sequence_lengths = batch_inputs.attention_mask.sum(dim=1).to("cpu")
        batch_length = sequence_lengths.shape[0]
        self.num_tokens += sequence_lengths.sum().item()

        # Resolve pooling indices for this batch
        pooling_indices = self._resolve_batch_pooling_indices(
            batch_inputs, batch_input_ids, sequence_lengths
        )

        # Run forward pass with gradient tracking disabled
        with torch.no_grad():
            if "intern" in self.model.__class__.__name__.lower():
                outputs = self.model.language_model(
                    **batch_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
            else:
                outputs = self.model(
                    **batch_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_states = outputs.hidden_states[1:]
            # remove the first layer (the embedding layer)
            hidden_states = [
                hidden_state.to(device=self.model.device, dtype=torch.bfloat16)
                for hidden_state in hidden_states
            ]

            # Detect internal image expansion (e.g. Cambrian: single -200 placeholder
            # internally becomes ~450 visual tokens, so hs_full_len > ids_full_len).
            # Validated by tinkering/understand_cambrian.py: V = hs_full_len - seq_len + 1,
            # visual block starts at img_pos_content (content-relative, already absolute in
            # expanded seq since Cambrian strips padding before expansion).
            is_expanded = (
                batch_input_ids is not None
                and hidden_states[0].shape[1] != batch_input_ids.shape[1]
            )

            if (
                is_expanded
                and self.tokens_pooling_method in _IMAGE_BASED_POOLING_METHODS
                and batch_length > 1
            ):
                raise RuntimeError(
                    f"'{self.tokens_pooling_method}' pooling with expansion models (e.g. Cambrian) "
                    f"requires batch_size=1. Got batch_length={batch_length}. "
                    "Set batch_size=1 in the dataloader."
                )

            batch_residual_stream = []
            for idx in range(batch_length):
                current_pooling_indices = None
                if pooling_indices is not None:
                    current_pooling_indices = pooling_indices[idx]

                if is_expanded and self.tokens_pooling_method in _IMAGE_BASED_POOLING_METHODS:
                    # Expansion-aware path: remap the single image placeholder to the full
                    # visual block in the expanded sequence.  All image-based pooling methods
                    # (mean_image, last_image, end_image, last10_image, last_img_and_next)
                    # receive the full visual_indices and apply their specific logic in
                    # _pool_tokens (e.g. [-1]+1 for end_image, [-10:] for last10_image).
                    seq_len_i = int(sequence_lengths[idx].item())
                    V = hidden_states[0].shape[1] - seq_len_i + 1
                    img_pos_content = int(current_pooling_indices[0].item())
                    visual_indices = torch.arange(img_pos_content, img_pos_content + V)
                    per_sample = torch.stack(
                        [hidden_states[layer_idx][idx] for layer_idx in range(self.num_layers)],
                        dim=0,
                    )  # (num_layers, hs_full_len, hidden_size)
                    result = self._pool_tokens(per_sample, pooling_indices=visual_indices)
                else:
                    per_sample = torch.stack(
                        [
                            hidden_states[layer_idx][idx, -sequence_lengths[idx] :, :]
                            for layer_idx in range(self.num_layers)
                        ],
                        dim=0,
                    )  # (num_layers, sequence_length(idx), hidden_size)
                    result = self._pool_tokens(
                        per_sample, pooling_indices=current_pooling_indices
                    )

                batch_residual_stream.append(
                    result.to(device="cpu", dtype=self.return_dtype)
                )

            if self.tokens_pooling_method != "none":
                # tensor of shape (num_layers, batch_length, hidden_size)
                batch_residual_stream = torch.stack(batch_residual_stream, dim=1)
                self._residual_stream[
                    :, self.counter : self.counter + batch_length, :
                ] = batch_residual_stream
                # tensor of shape (num_layers, num_samples, hidden_size)
            else:
                # dictionary of batch_length tensors of shape
                # (num_layers, sequence_length(idx), hidden_size)
                for idx in range(batch_length):
                    self._residual_stream[str(self.counter + idx)] = (
                        batch_residual_stream[idx]
                    )

            # Capture greedy next-token prediction from the last position's logits.
            # outputs.logits shape: (batch, seq_len, vocab_size) — computed for free
            # as part of the existing forward pass; no extra compute needed.
            if self.token_log is not None and self.tokenizer is not None:
                next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
                for nid in next_token_ids:
                    self.token_log.append(
                        self.tokenizer.decode(
                            [nid],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )

            # Per-layer logit-lens: project each layer's last-position hidden state through
            # the final norm + LM head.  hidden_states is outputs.hidden_states[1:] so
            # layer_idx=0 is the first transformer block — the raw embedding (index 0) is
            # never included.  Operates inside the existing no_grad context.
            if self.layer_token_log is not None:
                for idx in range(batch_length):
                    per_sample_tokens: list[str] = []
                    lm_dtype = next(self._lm_head.parameters()).dtype
                    for layer_idx in range(self.num_layers):
                        h = hidden_states[layer_idx][idx, -1:, :].to(lm_dtype)
                        if self._lm_norm is not None:
                            h = self._lm_norm(h)
                        logits = self._lm_head(h)                  # (1, vocab_size)
                        nid = int(logits[0].argmax().item())
                        per_sample_tokens.append(
                            self.tokenizer.decode(
                                [nid],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            )
                        )
                    self.layer_token_log.append(per_sample_tokens)

            del batch_residual_stream, hidden_states, outputs

        self.counter += batch_length


@deprecated(HOOK_BASED_TRACER_DEPRECATION_MSG)
class ResidualStreamPostMLPTracer(HookBasedResidualStreamTracer):
    """
    This class is used to extract the residual stream of a huggingface
    decoder-based multimodal model at the output of the MLP layer,
    before the residual connection.

    The residual stream can be pooled in different ways:
    - mean: average the residual stream over the sequence length
    - last: take the last token of the sequence
    - none: return the residual stream as is

    The residual stream is returned as a tensor of shape (num_layers, num_samples, hidden_size)
    if the tokens pooling method is not None, or as a dictionary of num_samples tensors of shape
    (num_layers, sequence_length(idx), hidden_size) if the tokens pooling method is None.
    """

    def __init__(
        self,
        model: ModelType,
        num_samples: int,
        processor: Optional[ProcessorType] = None,
        return_dtype: torch.dtype = torch.bfloat16,
        tokens_pooling_method: Optional[str] = None,
        embeddings_sampling_args: Optional[dict] = None,
    ):
        super().__init__(
            model=model,
            num_samples=num_samples,
            processor=processor,
            return_dtype=return_dtype,
            tokens_pooling_method=tokens_pooling_method,
            embeddings_sampling_args=embeddings_sampling_args,
        )
        print(
            f"\nThe estimated memory usage per token is {self.memory_usage_per_token:.2f} GB"
        )

    def _initialize_storage(self):
        """Initialize storage for post-MLP residual stream."""
        if self.tokens_pooling_method in [
            "mean",
            "last",
            "sample",
            "mean_image",
            "last_image",
            "mean_text_context",
            "last_caption",
            "last_user_text",
        ]:
            self._residual_stream = torch.zeros(
                self.num_layers,
                self.num_samples,
                self.hidden_size,
                device="cpu",
                dtype=self.return_dtype,
            )
        elif self.tokens_pooling_method == "none":
            self._residual_stream = {
                str(sample_idx): None for sample_idx in range(self.num_samples)
            }
            # each element is a tensor of shape (num_layers, hidden_size)

    def _initialize_none_pooling_storage(self, batch_length: int):
        """Initialize storage for None pooling method for post-MLP tracer."""
        for idx in range(batch_length):
            self._residual_stream[str(self.counter + idx)] = torch.zeros(
                self.num_layers,
                self.sequence_lengths[idx],
                self.hidden_size,
                device="cpu",
                dtype=self.return_dtype,
            )

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # output: (batch_size, max_sequence_length, hidden_size)
            x = output.to(torch.bfloat16).to(self.model.device)
            batch_length = x.shape[0]

            batch_residual_stream = []
            for idx in range(batch_length):
                # Get the position of the embedding to sample
                pooling_indices = None
                if self.current_batch_pooling_indices is not None:
                    pooling_indices = self.current_batch_pooling_indices[idx]

                batch_residual_stream.append(
                    self._pool_tokens(
                        # shape: (seq_len(idx), hidden_size_out)
                        x[idx, -self.sequence_lengths[idx] :, :],
                        pooling_indices=pooling_indices,
                    ).to(device="cpu", dtype=self.return_dtype)
                )
                # list of batch_length tensors of shape
                # (hidden_size,) or
                # (seq_len(idx), hidden_size) for layer_idx

            if self.tokens_pooling_method != "none":
                # tensor of shape: (batch, hidden_size)
                batch_residual_stream = torch.stack(batch_residual_stream, dim=0)
                start_idx = self.current_batch_start_idx
                end_idx = self.current_batch_start_idx + batch_length
                self._residual_stream[layer_idx, start_idx:end_idx, :] = (
                    batch_residual_stream
                )
                # tensor of shape: (batch, hidden_size) for layer_idx
            else:
                for idx in range(batch_length):
                    self._residual_stream[str(self.current_batch_start_idx + idx)][
                        layer_idx
                    ] = batch_residual_stream[idx]
                # dictionary of batch_length tensors of shape
                # (sequence_length(idx), hidden_size) for layer_idx

            del batch_residual_stream, x

        return hook

    def _trace(self):
        """Set up hooks for all target layers with pre-computation."""
        print("🔗 Setting up hooks...")
        for layer_idx in range(self.num_layers):
            mlp = self.model.language_model.model.layers[layer_idx].mlp

            # Register optimized hook
            hook = mlp.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(hook)


@deprecated(HOOK_BASED_TRACER_DEPRECATION_MSG)
class ResidualStreamHeadsProjectionTracer(HookBasedResidualStreamTracer):
    """
    This class is used to extract the residual stream of a huggingface
    decoder-based multimodal model at the output of each head of the
    multi-head attention layer, before the residual connection.

    The residual stream can be pooled in different ways:
    - mean: average the residual stream over the sequence length
    - last: take the last token of the sequence
    - none: return the residual stream as is

    The residual stream is returned as a tensor of shape
    (num_layers, num_samples, num_heads, hidden_size_out)
    if the tokens pooling method is not None, or as a dictionary of num_samples tensors of shape
    (num_layers, num_heads, sequence_length(idx), hidden_size_out) if the tokens pooling method is None.

    If the tokens pooling method is not None, the memory is bounded by
    O(num_layers * num_samples * num_heads * hidden_size_out * num_bytes).
    !For example, for a model with 32 layers, 4096 hidden size, 2500 samples, 32 heads,
    !and float16 dtype, the memory is bounded by 32 * 2500 * 32 * 4096 * 2 ~ 20 GB.

    !Using tokens_pooling_method=None is unlikely to be useful for this tracer and it is extremely memory intensive.
    """

    def __init__(
        self,
        model: ModelType,
        num_samples: int,
        processor: Optional[ProcessorType] = None,
        return_dtype: torch.dtype = torch.bfloat16,
        tokens_pooling_method: Optional[str] = None,
        embeddings_sampling_args: Optional[dict] = None,
    ):
        """
        Initialize heads projection tracer with additional attention head attributes.
        """
        # Set up head-specific attributes before calling parent
        self.num_heads = model.language_model.config.num_attention_heads
        self.head_dim = model.language_model.config.hidden_size // self.num_heads

        # Initialize parent class
        super().__init__(
            model=model,
            num_samples=num_samples,
            processor=processor,
            return_dtype=return_dtype,
            tokens_pooling_method=tokens_pooling_method,
            embeddings_sampling_args=embeddings_sampling_args,
        )

        # Override memory calculation for heads projection
        self.memory_usage_per_token = (
            self.num_layers
            * self.num_samples
            * self.num_heads
            * self.hidden_size
            * self._dtype_size(self.return_dtype)
            / 1024**3
        )
        print(
            f"\nThe estimated memory usage per token is {self.memory_usage_per_token:.2f} GB"
        )

        # Prevent memory issues
        if tokens_pooling_method == "none":
            raise ValueError(
                "⛔ Using tokens_pooling_method=None is extremely memory intensive for this tracer. "
                "Estimate memory usage by multiplying dataset tokens by memory usage per token."
            )

    def _initialize_storage(self):
        """Initialize storage for heads projection residual stream."""
        if self.tokens_pooling_method in [
            "mean",
            "last",
            "sample",
            "mean_image",
            "last_image",
            "mean_text_context",
            "last_caption",
            "last_user_text",
        ]:
            self._residual_stream = torch.zeros(
                self.num_layers,
                self.num_samples,
                self.num_heads,
                self.hidden_size,
                device="cpu",
                dtype=self.return_dtype,
            )
        elif self.tokens_pooling_method == "none":
            self._residual_stream = {
                str(sample_idx): None for sample_idx in range(self.num_samples)
            }
            # each element is a tensor of shape
            # (num_layers, num_heads, seq_len(idx), hidden_size_out)

        # shape: (num_layers, num_heads, head_dim, hidden_size_out)
        self._precomputed_weights = torch.zeros(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.hidden_size,
            device="cpu",
            dtype=self.return_dtype,
        )  # o_proj.weight.data for each layer

    def _initialize_none_pooling_storage(self, batch_length: int):
        """Initialize storage for None pooling method for heads projection tracer."""
        for idx in range(batch_length):
            self._residual_stream[str(self.counter + idx)] = torch.zeros(
                self.num_layers,
                self.num_heads,
                self.sequence_lengths[idx],
                self.hidden_size,
                device="cpu",
                dtype=self.return_dtype,
            )

    def _precompute_layer_info(self, layer_idx: int, module: torch.nn.Module):
        """Pre-compute weight slices and head indices for efficient processing."""

        # Pre-slice weight matrix for all heads
        W = module.weight.data.T  # (hidden_size_in, hidden_size_out)
        # For torch.nn.Linear, module.weight has shape
        # (hidden_size_out, hidden_size_in).
        # With W = module.weight.T, W has shape
        # (hidden_size_in, hidden_size_out), and the linear map is:
        # module(x) = x @ W (+b)
        # where x has shape (batch, max_seq_len, hidden_size_in)
        # and output has shape (batch, max_seq_len, hidden_size_out).
        # !W = module.weight.data could be taken but the code below to compute the projections
        # !should be adapted to use it.
        # ! In batched form: module(x) = x @ W (+b), where W = module.weight.T.
        # ! (Using module.weight directly would require rewriting the einsum accordingly.)

        # !To extract each head contribution to the input and project it to the residual stream,
        # !we need to slice the transposed weight matrix for each head, in blocks of
        # !(head_dim, hidden_size_out).
        W_heads = []
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = (h + 1) * self.head_dim
            W_head = W[start:end, :]  # (head_dim, hidden_size_out)
            W_heads.append(W_head)

        # shape: (num_heads, head_dim, hidden_size_out)
        self._precomputed_weights[layer_idx] = torch.stack(W_heads, dim=0)

    def _make_hook(self, layer_idx):
        """
        Make a hook for the forward pass of the model.
        """
        W_heads = self._precomputed_weights[layer_idx].to(
            device=self.model.device, dtype=torch.bfloat16
        )

        def hook(module, input, output):
            x = input[0]  # (batch, max_seq_len, hidden_size)
            batch_length = x.shape[0]

            heads_features = []
            for h in range(self.num_heads):
                start = h * self.head_dim
                end = (h + 1) * self.head_dim
                heads_features.append(x[:, :, start:end])
                # shape: (batch, max_seq_len, head_dim)
            x_heads = torch.stack(heads_features, dim=2).to(
                device=self.model.device, dtype=torch.bfloat16
            )

            # x_heads: (batch, max_seq_len, num_heads, head_dim)
            # W_heads: (num_heads, head_dim, hidden_size_out)
            projections = einsum(
                x_heads, W_heads, "b s n h, n h d_out -> b n s d_out"
            ).to(
                device=self.model.device, dtype=torch.bfloat16
            )  # projections: (batch, num_heads, max_seq_len, hidden_size_out)

            batch_projections = []
            for idx in range(batch_length):
                # Get the position of the embedding to sample
                pooling_indices = None
                if self.current_batch_pooling_indices is not None:
                    pooling_indices = self.current_batch_pooling_indices[idx]

                # shape: (num_heads, max_seq_len, hidden_size_out)
                # -> (num_heads, hidden_size_out) or
                # (num_heads, seq_len(idx), hidden_size_out)
                batch_projections.append(
                    self._pool_tokens(
                        projections[idx, :, -self.sequence_lengths[idx] :, :],
                        pooling_indices=pooling_indices,
                    ).to(device="cpu", dtype=self.return_dtype)
                )
                # list of batch_length tensors of shape
                # (num_heads, hidden_size_out) or
                # (num_heads, seq_len(idx), hidden_size_out) for layer_idx

            if self.tokens_pooling_method != "none":
                # tensor of shape: (batch, num_heads, hidden_size_out)
                batch_projections = torch.stack(batch_projections, dim=0)
                # Calculate the correct start index for this batch
                start_idx = self.current_batch_start_idx
                end_idx = self.current_batch_start_idx + batch_length
                # Assign each sample in the batch to consecutive indices
                self._residual_stream[layer_idx, start_idx:end_idx, :, :] = (
                    batch_projections
                )
            else:  # ! This branch is currently disabled
                # dictionary of batch_length tensors of shape
                # (num_heads, seq_len(idx), hidden_size_out) for layer_idx
                for idx in range(batch_length):
                    self._residual_stream[str(self.current_batch_start_idx + idx)][
                        layer_idx
                    ] = batch_projections[idx]

            del batch_projections, x_heads, projections, heads_features

        return hook

    def _trace(self):
        """Set up hooks for all target layers with pre-computation."""
        print("🔗 Setting up hooks...")
        for layer_idx in range(self.num_layers):
            o_proj = self.model.language_model.model.layers[layer_idx].self_attn.o_proj

            # Pre-compute layer-specific information
            self._precompute_layer_info(layer_idx, o_proj)

            # Register optimized hook
            hook = o_proj.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(hook)

        total_heads = self.num_heads * self.num_layers
        print(
            f"✅ Hooks registered for {total_heads} heads across {self.num_layers} layers"
        )

    def clear(self):
        """
        Clean up hooks and cached data, including precomputed weights.
        """
        if hasattr(self, "_precomputed_weights"):
            del self._precomputed_weights

        super().clear()


class ResidualStreamPostInputLayerNormTracer(HookBasedResidualStreamTracer):
    """Extract residual stream at the output of the input LayerNorm (pre-attention).

    Supports LLaVA 1.5, LLaVA OV, InternVL2, and Cambrian via architecture-aware
    hook registration. Intended for use with 'last' and 'last_user_text' pooling
    in pile-10k experiments; other pooling methods are structurally supported.
    """

    def _initialize_storage(self):
        if self.tokens_pooling_method in [
            "mean", "last", "sample", "mean_image", "last_image",
            "mean_text_context", "last_caption", "last_user_text",
        ]:
            self._residual_stream = torch.zeros(
                self.num_layers,
                self.num_samples,
                self.hidden_size,
                device="cpu",
                dtype=self.return_dtype,
            )
        elif self.tokens_pooling_method == "none":
            self._residual_stream = {str(i): None for i in range(self.num_samples)}

    def _initialize_none_pooling_storage(self, batch_length: int):
        for idx in range(batch_length):
            self._residual_stream[str(self.counter + idx)] = torch.zeros(
                self.num_layers,
                self.sequence_lengths[idx],
                self.hidden_size,
                device="cpu",
                dtype=self.return_dtype,
            )

    def _get_input_layernorm(self, layer_idx: int) -> torch.nn.Module:
        if is_cambrian_model(model=self.model):
            return self.model.model.layers[layer_idx].input_layernorm
        if "intern" in self.model.__class__.__name__.lower():
            return self.model.language_model.model.layers[layer_idx].attention_norm
        # LLaVA 1.5 (LlamaModel bare, .layers directly) and
        # LLaVA OV (LLaVAOneVision1_5_TextModel, .layers directly)
        return self.model.language_model.layers[layer_idx].input_layernorm

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # LayerNorm may return a tensor or (tensor, ...) depending on implementation.
            x = (output[0] if isinstance(output, tuple) else output).to(torch.bfloat16)
            batch_length = x.shape[0]
            batch_result = []
            for idx in range(batch_length):
                pi = (
                    self.current_batch_pooling_indices[idx]
                    if self.current_batch_pooling_indices is not None
                    else None
                )
                batch_result.append(
                    self._pool_tokens(
                        x[idx, -self.sequence_lengths[idx] :, :],
                        pooling_indices=pi,
                    ).to(device="cpu", dtype=self.return_dtype)
                )
            if self.tokens_pooling_method != "none":
                stacked = torch.stack(batch_result, dim=0)
                start = self.current_batch_start_idx
                self._residual_stream[layer_idx, start : start + batch_length, :] = stacked
            else:
                for idx in range(batch_length):
                    self._residual_stream[
                        str(self.current_batch_start_idx + idx)
                    ][layer_idx] = batch_result[idx]
            del batch_result, x

        return hook

    def _trace(self):
        print("🔗 Setting up input_layernorm hooks...")
        for layer_idx in range(self.num_layers):
            ln = self._get_input_layernorm(layer_idx)
            self.handles.append(ln.register_forward_hook(self._make_hook(layer_idx)))
        print(f"✅ Hooks registered for {self.num_layers} input_layernorm layers")
        print(
            f"\nThe estimated memory usage per token is {self.memory_usage_per_token:.2f} GB"
        )


def residual_stream_tracer(
    model: ModelType,
    processed_dataloader: DataLoader,
    processor: Optional[ProcessorType] = None,
    return_dtype: torch.dtype = torch.bfloat16,
    tokens_pooling_method: Optional[
        Literal[
            "mean",
            "last",
            "none",
            "sample",
            "mean_image",
            "last_image",
            "mean_text_context",
            "last_caption",
            "last_user_text",
        ]
    ] = "last",
    residual_stream_type: Literal[
        "output_layer", "post_mlp", "heads_projection"
    ] = "output_layer",
    return_deepcopy: bool = True,
    embeddings_sampling_args: Optional[dict] = None,
    save_path: Optional[str] = None,
    save_representations: bool = False,
    save_metadata: Optional[Dict] = None,
    debug: bool = False,
    token_log: Optional[List[str]] = None,
    layer_token_log: Optional[List[List[str]]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Extract the residual stream of a huggingface multimodal decoder-based,
    based on the LLaVA architecture, from the dataloader.

    The residual stream can be extracted at three different locations:
    - the output of a decoder layer,
    - the output of the MLP layer, before the residual connection, or
    - the output of each head of the multi-head attention layer, before the
    residual connection.

    The residual stream can be pooled in different ways:
    - mean: average the residual stream over the sequence length
    - last: take the last token of the sequence
    - none: return the residual stream as is
    - sample: sample specific positions (e.g. image or text token)
    - mean_image: average over all image tokens

    If the residual stream is extracted at the output of a decoder layer or at the output of the MLP layer,
    the residual stream is returned as a tensor of shape (num_layers, num_samples, hidden_size) if the
    tokens pooling method is not None, or as a dictionary of num_samples tensors of shape
    (num_layers, sequence_length(idx), hidden_size) if the tokens pooling method is None.

    If the residual stream is extracted at the output of each head of the multi-head attention layer,
    before the residual connection, the residual stream is returned as a tensor of shape
    (num_layers, num_samples, num_heads, hidden_size) if the tokens pooling method is not None, or as a
    dictionary of num_samples tensors of shape (num_layers, num_heads, sequence_length(idx), hidden_size)
    if the tokens pooling method is None.

    Args:
        model (ModelType): huggingface multimodal decoder-based model
        processed_dataloader (DataLoader): processed dataloader to iterate over the dataset
        return_dtype (torch.dtype, optional): dtype of the residual stream. Defaults to torch.bfloat16.
        tokens_pooling_method (Optional[Literal["mean", "last", "none", "sample", "mean_image", "last_image", "mean_text_context", "last_caption"]], optional): method to pool the tokens. Defaults to "last".
            Note: ``mean_context``, ``mean_image_last_caption``, and ``last_image_last_caption`` are
            *virtual* pooling methods dispatched at the experiment-script level.
            ``mean_context`` splits into ``mean_image`` (images) and ``mean_text_context`` (text).
            ``mean_image_last_caption`` splits into ``mean_image`` (images) and ``last_caption`` (text).
            ``last_image_last_caption`` splits into ``last_image`` (images) and ``last_caption`` (text).
            Pass the concrete method here.
        residual_stream_type (str, optional): type of the residual stream. Defaults to "output_layer".
        return_deepcopy (bool, optional): whether to return a deepcopy of the residual stream.
            Defaults to True, which deletes the tracer and intermediate variables.
            If False, the residual stream is returned as is, and the tracer is not deleted.
        embeddings_sampling_args (Optional[dict], optional): arguments for sampling embeddings from specific positions.
            If provided, this overrides tokens_pooling_method and samples embeddings from specific image/text positions.
            Expected keys: 'image_token_id', 'image_seq_length', 'pad_token_id', 'text_direction',
            'skip_image_pos', 'skip_text_pos', 'generator'.
        save_path (str, optional): path to save the extracted representation. If provided and file exists,
            it will be loaded instead of computed (unless valid data is not found).
        save_representations (bool, optional): whether to save the extracted representations to `save_path`.
            Defaults to False.
        save_metadata (dict, optional): metadata to save along with the representations.

    Returns:
        Union[torch.Tensor, Dict[str, torch.Tensor]]: residual stream of the model

    Raises:
        ValueError: If residual_stream_type or tokens_pooling_method are invalid
        ValueError: If processed_dataloader is empty
    """

    # 1. Check for cached representation - Logic moved inside to avoid redundant computations
    if save_path and os.path.exists(save_path) and not save_representations:
        # Note: If save_representations is True, we assume we want to OVERWRITE or recompute?
        # User prompt: "check if the representations already exists to avoid re-doing computations"
        # "check if the representations already exists to avoid re-doing computations... and later... save representations that by default is set to `false`"
        # So if exists, load it. If doesn't exist, compute. If compute finishes AND save=True, save it.
        # But if user set save=True, and it exists, maybe they want to verify?
        # Standard caching behavior: if exists, use it.
        print(f"  ✅ Found existing cache at: {save_path}")
        try:
            # We explicitly load to CPU
            loaded = load_file(save_path)
            if "residual_stream" in loaded:
                return loaded["residual_stream"]
            # Should have only one key if saved by us as tensor
            return next(iter(loaded.values()))
        except Exception as e:
            print(f"  ⚠️ Error loading cache: {e}. Recomputing...")

    # Parameter validation
    if residual_stream_type not in VALID_RESIDUAL_STREAM_TYPES:
        raise ValueError(
            f"Invalid residual_stream_type '{residual_stream_type}'. "
            f"Must be one of: {VALID_RESIDUAL_STREAM_TYPES}"
        )

    if is_cambrian_model(model=model) and residual_stream_type in [
        "post_mlp", "heads_projection"
    ]:
        raise ValueError(
            "Cambrian supports only residual_stream_type='output_layer' or "
            "'post_input_layernorm'. "
            f"Received: '{residual_stream_type}'."
        )

    if tokens_pooling_method not in VALID_TOKENS_POOLING_METHODS:
        raise ValueError(
            f"Invalid tokens_pooling_method '{tokens_pooling_method}'. "
            f"Must be one of: {VALID_TOKENS_POOLING_METHODS}"
        )

    # Extract dataset info with validation
    dataset_size = len(processed_dataloader.dataset)

    if dataset_size == 0:
        raise ValueError(
            "Dataset is empty. Cannot extract residual stream from empty dataset."
        )

    print("\n" + "-" * 80)
    print(
        f"🔍 Tracing residual stream: {residual_stream_type} with {tokens_pooling_method} pooling"
    )
    print("-" * 80)

    # Initialize the appropriate tracer based on residual_stream_type
    tracer_kwargs = {
        "model": model,
        "num_samples": dataset_size,
        "processor": processor,
        "tokens_pooling_method": tokens_pooling_method,
        "return_dtype": return_dtype,
        "embeddings_sampling_args": embeddings_sampling_args,
    }

    if residual_stream_type == "output_layer":
        tracer = ResidualStreamOutputLayerTracer(
            **tracer_kwargs, token_log=token_log, layer_token_log=layer_token_log
        )
    elif residual_stream_type == "post_mlp":
        tracer = ResidualStreamPostMLPTracer(**tracer_kwargs)
    elif residual_stream_type == "heads_projection":
        tracer = ResidualStreamHeadsProjectionTracer(**tracer_kwargs)
    elif residual_stream_type == "post_input_layernorm":
        tracer = ResidualStreamPostInputLayerNormTracer(**tracer_kwargs)

    # Initialize progress tracking variables
    samples_processed = 0
    update_count = 0
    update_every = max(1, int(0.1 * dataset_size))

    # InternVL-specific: get image context token id for combined embeddings
    if "intern" in model.__class__.__name__.lower():
        if processor is None:
            raise ValueError(
                "Processor is required for InternVL models to get image context token ID."
            )
        img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    print(f"📊 Processing {dataset_size} samples...")
    progress_bar = tqdm(
        total=dataset_size,
        desc="Extracting residual stream",
        unit="sample",
    )

    # Start timing
    start_time = time.time()

    # Process batches
    for batch in processed_dataloader:
        batch_input_ids = None
        cambrian_images = None
        cambrian_image_sizes = None

        # Prepare batch_inputs for tracing (batch stays untouched)
        if isinstance(batch, (list, tuple)):
            if "llava" in model.__class__.__name__.lower():
                batch_inputs = batch[0]
                batch_input_ids = batch_inputs["input_ids"]

            elif "intern" in model.__class__.__name__.lower():
                if len(batch) > 1 and isinstance(batch[1], torch.Tensor):
                    # Image mode: combine text + vision embeddings
                    pixel_values = batch[1]
                    tokenized = batch[0]
                    input_ids = tokenized["input_ids"]
                    attention_mask = tokenized["attention_mask"]
                    batch_input_ids = input_ids
                    combined_embeds = get_combined_embeddings(
                        model,
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        img_context_token_id=img_context_token_id,
                    )
                    batch_inputs = BatchFeature(
                        {
                            "inputs_embeds": combined_embeds,
                            "attention_mask": attention_mask,
                        }
                    )
                else:
                    # Text-only mode: use input_ids directly
                    batch_inputs = batch[0]
                    batch_input_ids = batch_inputs["input_ids"]

            elif is_cambrian_model(model=model):
                # Cambrian batch format from collate functions:
                #   Multimodal: (tokenized, image_tensors, image_sizes, prompts)
                #   Text-only:  (tokenized, prompts)
                tokenized = batch[0]
                batch_input_ids = tokenized["input_ids"]
                cambrian_images = None
                cambrian_image_sizes = None
                if len(batch) >= 4 and isinstance(batch[1], (list, torch.Tensor)):
                    cambrian_images = batch[1]
                    cambrian_image_sizes = batch[2]
                batch_inputs = tokenized

            else:
                batch_inputs = batch[0]
                if "input_ids" in batch_inputs:
                    batch_input_ids = batch_inputs["input_ids"]
        else:
            batch_inputs = batch
            if "input_ids" in batch_inputs:
                batch_input_ids = batch_inputs["input_ids"]

        batch_inputs = batch_inputs.to(model.device)

        # Inject Cambrian vision data after .to() (list[Tensor] needs manual move)
        if is_cambrian_model(model=model) and cambrian_images is not None:
            if isinstance(cambrian_images, list):
                batch_inputs["images"] = [
                    t.to(device=model.device, dtype=torch.float16)
                    for t in cambrian_images
                ]
            else:
                batch_inputs["images"] = cambrian_images.to(
                    device=model.device, dtype=torch.float16
                )
            batch_inputs["image_sizes"] = cambrian_image_sizes

        # ------------------------------------------------------------------
        # Debug: generate and print model responses
        # ------------------------------------------------------------------
        if debug and "intern" in model.__class__.__name__.lower():
            # Extract queries for debugging (always last element for InternVL)
            # !provided that the dataloader doesn't explicitly return the answers
            # !which are also a list at the end of the batch
            # !the use of this debugging feature is limited the use with the
            # !multimodality_similarity.py script
            queries = None
            if isinstance(batch, (list, tuple)):
                if len(batch) > 1 and isinstance(batch[-1], list):
                    queries = batch[-1]

            print(f"batch type: {type(batch)} with length {len(batch)}")
            if "input_ids" in batch_inputs:
                responses = generate_responses(
                    model,
                    processor,
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                )
            else:
                responses = generate_responses(
                    model,
                    processor,
                    inputs_embeds=batch_inputs["inputs_embeds"],
                    attention_mask=batch_inputs["attention_mask"],
                )

            for idx, response in enumerate(responses):
                if queries is not None:
                    query = queries[idx]
                    query = query.replace("<IMG_CONTEXT>", "")
                    query = query.replace("</img>", "").replace("<img>", "<image>")
                    print(f"Query: {query}\n   Response: {response}\n")
                else:
                    print(f"Response: {response}\n")
                print("-" * 40)

        if debug and is_cambrian_model(model=model):
            # Cambrian debug: prompts are the last list element in the batch
            prompts = None
            if isinstance(batch, (list, tuple)) and isinstance(batch[-1], list):
                prompts = batch[-1]

            generate_kwargs = {
                "inputs": batch_inputs["input_ids"],
                "attention_mask": batch_inputs["attention_mask"],
                "max_new_tokens": 50,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": False,
            }
            if "images" in batch_inputs:
                generate_kwargs["images"] = batch_inputs["images"]
                generate_kwargs["image_sizes"] = batch_inputs["image_sizes"]

            with torch.inference_mode():
                output_ids = model.generate(**generate_kwargs)

            prompt_len = batch_inputs["input_ids"].shape[1]
            for idx in range(output_ids.shape[0]):
                generated = output_ids[idx, prompt_len:]
                response = processor.decode(generated, skip_special_tokens=True).strip()
                if prompts is not None:
                    print(f"Prompt: {prompts[idx]}\n   Response: {response}\n")
                else:
                    print(f"Response: {response}\n")
                print("-" * 40)
        # ------------------------------------------------------------------

        current_batch_size = batch_inputs.attention_mask.shape[0]
        tracer.trace_batch(batch_inputs, batch_input_ids=batch_input_ids)
        del batch, batch_inputs

        # Update progress tracking
        samples_processed += current_batch_size
        update_count += current_batch_size

        # Update progress bar and clean memory periodically
        if update_count % update_every == 0 or samples_processed == dataset_size:
            progress_bar.update(update_count)
            torch.cuda.empty_cache()
            gc.collect()
            update_count = 0

    progress_bar.close()
    end_time = time.time()

    # Extract results and cleanup
    residual_stream = tracer.get_residual_stream(return_deepcopy=return_deepcopy)
    tracer.clear()
    if return_deepcopy:
        del tracer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "-" * 80)
    print(
        f"✅ Residual stream extracted successfully in ⌛ {(end_time - start_time) / 60:.2f} minutes"
    )
    print("-" * 80 + "\n")

    # 2. Save representation if requested
    if save_representations and save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"  💾 Saving representation to: {save_path}")

        # Prepare metadata (flat dict of strings)
        valid_metadata = {}
        if save_metadata:
            for k, v in save_metadata.items():
                valid_metadata[k] = str(v)

        try:
            if isinstance(residual_stream, torch.Tensor):
                save_file(
                    {"residual_stream": residual_stream},
                    save_path,
                    metadata=valid_metadata,
                )
            else:
                # Case for 'none' pooling which returns dict of tensors
                save_file(residual_stream, save_path, metadata=valid_metadata)
        except Exception as e:
            print(f"  ❌ Error saving representation: {e}")

    return residual_stream
