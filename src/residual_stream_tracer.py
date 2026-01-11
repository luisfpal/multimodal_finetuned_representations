import abc
import copy
import gc
import time
from typing import Dict, Optional, Union, Literal
import torch
from einops import einsum
from tqdm import tqdm
from rich import print
from transformers.feature_extraction_utils import BatchFeature
from torch.utils.data import DataLoader
from utils import ModelType, sample_image_and_text_positions

# Constants for valid parameter values
VALID_RESIDUAL_STREAM_TYPES = ["output_layer", "post_mlp", "heads_projection"]
VALID_TOKENS_POOLING_METHODS = ["mean", "last", "none", "sample"]


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
        return_dtype: torch.dtype = torch.float16,
        tokens_pooling_method: Optional[str] = "last",
        embeddings_sampling_args: Optional[dict] = None,
    ):
        """
        Args:
            model (ModelType): huggingface multimodal decoder-based model
            num_samples (int): number of samples to trace, e.g. number of samples in the dataset
            return_dtype (torch.dtype, optional): dtype of the residual stream. Defaults to torch.float16.
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
        self.model = model
        self.tokens_pooling_method = tokens_pooling_method
        self.embeddings_sampling_args = embeddings_sampling_args
        self.num_layers = self.model.language_model.config.num_hidden_layers
        self.hidden_size = self.model.language_model.config.hidden_size
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
            "Precision depends on dtype. torch.float32 gives high precision but is memory intensive. "
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
            if (
                self.embeddings_sampling_args.get("skip_image_pos")
                and self.embeddings_sampling_args.get("skip_text_pos")
            ):
                raise ValueError("Cannot skip both image and text positions")
            elif (
                not self.embeddings_sampling_args.get("skip_image_pos")
                and not self.embeddings_sampling_args.get("skip_text_pos")
            ):
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
        positions = sample_image_and_text_positions(
            input_ids=self.current_batch_input_ids,
            **self.embeddings_sampling_args,
        )

        # Two positions per batch element (image and text)
        batch_indices = torch.arange(
            hidden_states.shape[0], device=hidden_states.device
        )
        # Take the first position (image) if both are sampled, otherwise use the single position
        return hidden_states[batch_indices, positions]

    def _pool_tokens(
        self,
        hidden_states: torch.Tensor,
        embedding_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool the tokens of the hidden states, one sample at a time.

        Args:
            hidden_states: Input tensor to pool
            embedding_position: Position of the embedding to sample.
        Returns:
            Pooled tensor or original tensor if no pooling
        """
        if self.tokens_pooling_method == "sample" and embedding_position is None:
            raise ValueError("embedding_position must be provided when tokens_pooling_method is 'sample'")
        ndims = hidden_states.ndim
        if self.tokens_pooling_method == "mean":
            if ndims == 3:
                return hidden_states.mean(dim=1) # (layers, hidden_size)
            elif ndims == 2:
                return hidden_states.mean(dim=0) # (hidden_size)
            else:
                raise ValueError(f"Unsupported tensor dimensions for mean pooling: {ndims}")
        elif self.tokens_pooling_method == "last":
            # Handle different tensor shapes for last token pooling
            if ndims == 2:
                return hidden_states[-1, :]
            elif ndims == 3:
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
                return hidden_states[:, embedding_position, :]
            # case 2: (seq_len(idx), hidden_size)
            elif ndims == 2:
                return hidden_states[embedding_position, :]
            else:
                raise ValueError(f"Unsupported tensor dimensions for sampling: {hidden_states.ndim}")
        else:
            raise ValueError(
                f"Invalid tokens pooling method: {self.tokens_pooling_method}"
            )

    @abc.abstractmethod
    def trace_batch(self, batch_inputs: BatchFeature):
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
            return copy.deepcopy(self._residual_stream)
        else:
            return self._residual_stream

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
        return_dtype: torch.dtype = torch.float16,
        tokens_pooling_method: Optional[str] = None,
        embeddings_sampling_args: Optional[dict] = None,
    ):
        """
        Initialize hook-based tracer with additional hook management attributes.
        """
        # Initialize parent class first
        super().__init__(
            model,
            num_samples,
            return_dtype,
            tokens_pooling_method,
            embeddings_sampling_args,
        )

        # Hook-specific attributes
        self.handles = []
        self.current_batch_start_idx = 0
        self.sequence_lengths = None
        self.current_batch_embeddings_positions = None

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

    def trace_batch(self, batch_inputs: BatchFeature):
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

            # Store input_ids for embeddings sampling if needed
            if self.tokens_pooling_method == "sample":
                self.current_batch_embeddings_positions = sample_image_and_text_positions(
                    input_ids=batch_inputs.input_ids,
                    **self.embeddings_sampling_args,
                )

            # Increment counter for next batch
            self.counter += batch_length
            _ = self.model(**batch_inputs)

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
        if hasattr(self, "current_batch_embeddings_positions"):
            del self.current_batch_embeddings_positions

        super().clear()


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
        return_dtype: torch.dtype = torch.float16,
        tokens_pooling_method: Optional[str] = None,
        embeddings_sampling_args: Optional[dict] = None,
    ):
        super().__init__(
            model,
            num_samples,
            return_dtype,
            tokens_pooling_method,
            embeddings_sampling_args,
        )
        print(
            f"\nThe estimated memory usage per token is {self.memory_usage_per_token:.2f} GB"
        )

    def _initialize_storage(self):
        """Initialize storage for output residual stream."""
        if self.tokens_pooling_method in ["mean", "last", "sample"]:
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

    def trace_batch(self, batch_inputs: BatchFeature):
        """
        Trace a tokenized/processed batch of inputs of samples along all the layers of the model.
        """
        sequence_lengths = batch_inputs.attention_mask.sum(dim=1).to("cpu")
        batch_length = sequence_lengths.shape[0]
        self.num_tokens += sequence_lengths.sum().item()
        
        # Store input_ids for embeddings sampling if needed
        embeddings_positions = None
        if self.tokens_pooling_method == "sample":
            embeddings_positions = sample_image_and_text_positions(
                input_ids=batch_inputs.input_ids,
                **self.embeddings_sampling_args,
            ) # (B,)

        # Run forward pass with gradient tracking disabled
        with torch.no_grad():
            outputs = self.model(**batch_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]
            # remove the first layer (the embedding layer)
            hidden_states = [
                hidden_state.to(device=self.model.device, dtype=torch.float32)
                for hidden_state in hidden_states
            ]
            # list of num_hidden_layers tensors of shape (batch_size, max_sequence_length, hidden_size)

            batch_residual_stream = []
            for idx in range(batch_length):
                # Get the position of the embedding to sample
                if self.tokens_pooling_method == "sample":
                    embedding_position = embeddings_positions[idx]
                else:
                    embedding_position = None
                batch_residual_stream.append(
                    self._pool_tokens(
                        torch.stack(
                            [  # num_layers tensors of shape (sequence_length(idx), hidden_size)
                                hidden_states[layer_idx][
                                    idx, -sequence_lengths[idx] :, :
                                ]
                                for layer_idx in range(self.num_layers)
                            ],
                            dim=0,
                        ),  # shape: (num_layers, sequence_length(idx), hidden_size)
                        embedding_position=embedding_position,
                    ).to(device="cpu", dtype=self.return_dtype)
                    # batch_length list of tensors of shape
                    # (num_layers, sequence_length(idx), hidden_size) or
                    # (num_layers, hidden_size)
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

            del batch_residual_stream, hidden_states, outputs

        self.counter += batch_length


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
        return_dtype: torch.dtype = torch.float16,
        tokens_pooling_method: Optional[str] = None,
        embeddings_sampling_args: Optional[dict] = None,
    ):
        super().__init__(
            model,
            num_samples,
            return_dtype,
            tokens_pooling_method,
            embeddings_sampling_args,
        )
        print(
            f"\nThe estimated memory usage per token is {self.memory_usage_per_token:.2f} GB"
        )

    def _initialize_storage(self):
        """Initialize storage for post-MLP residual stream."""
        if self.tokens_pooling_method in ["mean", "last", "sample"]:
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
            x = output.to(torch.float32).to(self.model.device)
            batch_length = x.shape[0]

            batch_residual_stream = []
            for idx in range(batch_length):
                # Get the position of the embedding to sample
                if self.tokens_pooling_method == "sample":
                    embedding_position = self.current_batch_embeddings_positions[idx]
                else:
                    embedding_position = None
                batch_residual_stream.append(
                    self._pool_tokens(
                        # shape: (seq_len(idx), hidden_size_out)
                        x[idx, -self.sequence_lengths[idx] :, :],
                        embedding_position=embedding_position,
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
        return_dtype: torch.dtype = torch.float16,
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
            model,
            num_samples,
            return_dtype,
            tokens_pooling_method,
            embeddings_sampling_args,
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
        if self.tokens_pooling_method in ["mean", "last", "sample"]:
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
        # The application of the linear layer with no bias is:
        # module(x) = x * W.T
        # where x is the input tensor of shape (batch, max_seq_len, hidden_size_in)
        # and W.T is the transposed weight matrix of shape (hidden_size_in, hidden_size_out)
        # The output tensor of shape (batch, max_seq_len, hidden_size_out)
        # !W = module.weight.data could be taken but the code below to compute the projections
        # !should be adapted to use it.
        # ! module(x) = x * W.T = W * x (+b)

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
            device=self.model.device, dtype=torch.float32
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
                device=self.model.device, dtype=torch.float32
            )

            # x_heads: (batch, max_seq_len, num_heads, head_dim)
            # W_heads: (num_heads, head_dim, hidden_size_out)
            projections = einsum(
                x_heads, W_heads, "b s n h, n h d_out -> b n s d_out"
            ).to(
                device=self.model.device, dtype=torch.float32
            )  # projections: (batch, num_heads, max_seq_len, hidden_size_out)

            batch_projections = []
            for idx in range(batch_length):
                # Get the position of the embedding to sample
                if self.tokens_pooling_method == "sample":
                    embedding_position = self.current_batch_embeddings_positions[idx]
                else:
                    embedding_position = None
                
                # shape: (num_heads, max_seq_len, hidden_size_out)
                # -> (num_heads, hidden_size_out) or
                # (num_heads, seq_len(idx), hidden_size_out)
                batch_projections.append(
                    self._pool_tokens(
                        projections[idx, :, -self.sequence_lengths[idx] :, :],
                        embedding_position=embedding_position,
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


def residual_stream_tracer(
    model: ModelType,
    processed_dataloader: DataLoader,
    return_dtype: torch.dtype = torch.float16,
    tokens_pooling_method: Optional[Literal["mean", "last", "none", "sample"]] = "last",
    residual_stream_type: Literal[
        "output_layer", "post_mlp", "heads_projection"
    ] = "output_layer",
    return_deepcopy: bool = True,
    embeddings_sampling_args: Optional[dict] = None,
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
        return_dtype (torch.dtype, optional): dtype of the residual stream. Defaults to torch.float16.
        tokens_pooling_method (Optional[Literal["mean", "last", "none", "sample"]], optional): method to pool the tokens. Defaults to "last".
        residual_stream_type (str, optional): type of the residual stream. Defaults to "output_layer".
        return_deepcopy (bool, optional): whether to return a deepcopy of the residual stream.
            Defaults to True, which deletes the tracer and intermediate variables.
            If False, the residual stream is returned as is, and the tracer is not deleted.
        embeddings_sampling_args (Optional[dict], optional): arguments for sampling embeddings from specific positions.
            If provided, this overrides tokens_pooling_method and samples embeddings from specific image/text positions.
            Expected keys: 'image_token_id', 'image_seq_length', 'pad_token_id', 'text_direction',
            'skip_image_pos', 'skip_text_pos', 'generator'.

    Returns:
        Union[torch.Tensor, Dict[str, torch.Tensor]]: residual stream of the model

    Raises:
        ValueError: If residual_stream_type or tokens_pooling_method are invalid
        ValueError: If processed_dataloader is empty
    """

    # Parameter validation
    if residual_stream_type not in VALID_RESIDUAL_STREAM_TYPES:
        raise ValueError(
            f"Invalid residual_stream_type '{residual_stream_type}'. "
            f"Must be one of: {VALID_RESIDUAL_STREAM_TYPES}"
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
        "tokens_pooling_method": tokens_pooling_method,
        "return_dtype": return_dtype,
        "embeddings_sampling_args": embeddings_sampling_args,
    }

    if residual_stream_type == "output_layer":
        tracer = ResidualStreamOutputLayerTracer(**tracer_kwargs)
    elif residual_stream_type == "post_mlp":
        tracer = ResidualStreamPostMLPTracer(**tracer_kwargs)
    elif residual_stream_type == "heads_projection":
        tracer = ResidualStreamHeadsProjectionTracer(**tracer_kwargs)

    # Initialize progress tracking variables
    samples_processed = 0
    update_count = 0
    update_every = max(1, int(0.1 * dataset_size))

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
        batch = batch.to(model.device)
        current_batch_size = batch.attention_mask.shape[0]
        tracer.trace_batch(batch)
        del batch

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

    return residual_stream
