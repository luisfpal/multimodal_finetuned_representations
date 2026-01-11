import copy
from typing import List
import torch
from .model_utils import ModelType


def layers_list_patching(
    source_model: ModelType,
    target_model: ModelType,
    layers: List[int],
    in_place_patching: bool = False,
) -> ModelType:
    """
    Core patching function: patch specific layers from source to target model.

    Args:
        source_model: Model to copy layers from
        target_model: Model to copy layers to
        layers: List of layer indices to patch (use num_layers to include lm_head)
        in_place_patching: If True, modify target_model in-place; if False, return deep copy

    Returns:
        Target model with patched layers (deep copy on CPU if in_place_patching=False)

    Model structure (source_model and target_model are the same):
        - source_model.language_model
            - model (applied in the following order)
                - embed_tokens: nn.Embedding
                - rotary_emb: LlamaRotaryEmbedding: has no parameters
                - layers: nn.ModuleList
                - norm: LlamaRMSNorm
            - lm_head
    """

    print(f"Patching layers {layers} from source model to target model... 🔄")

    if not layers:
        raise ValueError("layers parameter must contain at least one layer index")

    if not in_place_patching:
        # Create deep copy of target model and move to CPU
        target_model_copy = copy.deepcopy(target_model).cpu()
    else:
        target_model_copy = target_model

    num_layers = source_model.language_model.config.num_hidden_layers

    # Validate layer indices
    for layer_idx in layers:
        if not (0 <= layer_idx <= num_layers):  # num_layers is valid (means lm_head)
            raise ValueError(
                f"Layer index {layer_idx} must be in range [0, {num_layers}]"
            )

    # Get layer references
    source_layers = source_model.language_model.model.layers
    target_layers = target_model_copy.language_model.model.layers

    with torch.no_grad():
        # Patch the lm_head if num_layers is in the list
        sorted_layers = sorted(layers)

        # Handle lm_head patching (represented by num_layers index)
        if sorted_layers and num_layers == sorted_layers[-1]:
            # Patch lm_head weights
            target_model_copy.language_model.lm_head.load_state_dict(
                source_model.language_model.lm_head.state_dict()
            )
            # Remove num_layers from the list (it's not an actual layer index)
            sorted_layers.pop()

        if not sorted_layers:
            return target_model_copy

        # Patch the specified layers
        for layer_idx in sorted_layers:
            target_layers[layer_idx].load_state_dict(
                source_layers[layer_idx].state_dict()
            )

        # Patch the embedding layer if the first layer is patched
        if sorted_layers and sorted_layers[0] == 0:
            target_model_copy.language_model.model.embed_tokens.load_state_dict(
                source_model.language_model.model.embed_tokens.state_dict()
            )

        # Patch the layer norm if the last layer is patched
        if sorted_layers and sorted_layers[-1] == num_layers - 1:
            target_model_copy.language_model.model.norm.load_state_dict(
                source_model.language_model.model.norm.state_dict()
            )

    return target_model_copy


def sliding_window_patching(
    source_model: ModelType,
    target_model: ModelType,
    start_layer: int = 0,
    window_size: int = 1,
    in_place_patching: bool = False,
) -> ModelType:
    """
    Helper function to patch layers weights from source_model to target_model
    in a sliding window manner, i.e., groups of layers are copied
    from a source model to a target model.

    Args:
        source_model: Model to copy layers from
        target_model: Model to copy layers to
        start_layer: First layer index to copy
        window_size: Number of consecutive layers to copy
        in_place_patching: If True, modify target_model in-place; if False, return deep copy

    Returns:
        Target model with patched layers (deep copy on CPU if in_place_patching=False)
    """

    num_layers = source_model.language_model.config.num_hidden_layers
    end_layer = start_layer + window_size

    # Validate bounds
    if not (0 <= start_layer < num_layers):
        raise ValueError(
            f"start_layer {start_layer} must be in range [0, {num_layers}), including the lm_head"
        )
    if end_layer > num_layers:
        raise ValueError(
            "Window extends beyond model: "
            f"final layer index ({end_layer}) > number of layers + 1 ({num_layers})"
        )

    # Generate list of layers to patch
    layers_to_patch = list(range(start_layer, end_layer))

    # Include lm_head if window reaches the end
    if end_layer == num_layers:
        layers_to_patch.append(num_layers)  # num_layers signals lm_head patching

    # Use the core patching function
    return layers_list_patching(
        source_model=source_model,
        target_model=target_model,
        layers=layers_to_patch,
        in_place_patching=in_place_patching,
    )


def cumulative_patching(
    source_model: ModelType,
    target_model: ModelType,
    split_layer: int,
    in_place_patching: bool = False,
    reverse_cumulative_layers: bool = False,
) -> ModelType:
    """
    Split target model: copy first part from source, keep second part unchanged.

    Args:
        source_model: Model to copy layers from
        target_model: Model to modify
        split_layer: Layer index where to split (0 to split_layer-1 from source)
        in_place_patching: If True, modify target_model in-place; if False, return deep copy
        reverse_cumulative_layers: If True, patch layers cumulatively from end to start;
                                   if False, patch layers cumulatively from start to split_layer

    Returns:
        Target model with patched layers (deep copy on CPU if in_place_patching=False)
    """

    num_layers = source_model.language_model.config.num_hidden_layers

    # Validate bounds
    if not (0 <= split_layer <= num_layers):
        raise ValueError(
            f"split_layer {split_layer} must be in range [0, {num_layers}]"
        )

    # Generate list of layers to patch
    start_layer = 0 if not reverse_cumulative_layers else num_layers
    end_layer = split_layer + 1 if not reverse_cumulative_layers else split_layer
    tuple_layers = (start_layer, end_layer, 1 if not reverse_cumulative_layers else -1)
    layers_to_patch = list(range(*tuple_layers))

    # Include lm_head if splitting at the very end
    if split_layer == num_layers:
        layers_to_patch.append(num_layers)  # num_layers signals lm_head patching

    # Use the core patching function
    return layers_list_patching(
        source_model=source_model,
        target_model=target_model,
        layers=layers_to_patch,
        in_place_patching=in_place_patching,
    )


def patch_layers_weights(
    source_model: ModelType,
    target_model: ModelType,
    method: str,
    start_layer: int = 0,
    window_size: int = 1,
    split_layer: int = 16,
    layers: List[int] = None,
    in_place_patching: bool = False,
    reverse_cumulative_layers: bool = False,
) -> ModelType:
    """
    Patch layers between models using specified method.

    Args:
        source_model: Model to copy from
        target_model: Model to copy to
        method: Either 'sliding_window', 'cumulative', or 'layers_list'
        start_layer: Starting layer for sliding_window method
        window_size: Window size for sliding_window method
        split_layer: Split point for cumulative method
        layers: List of layer indices for layers_list method
        in_place_patching: If True, modify target_model in-place; if False, return deep copy
        reverse_cumulative_layers: If True, patch layers cumulatively from end to start (cumulative only)

    Returns:
        Target model with patched layers (deep copy on CPU if in_place_patching=False)
    """
    if method == "sliding_window":
        return sliding_window_patching(
            source_model=source_model,
            target_model=target_model,
            start_layer=start_layer,
            window_size=window_size,
            in_place_patching=in_place_patching,
        )
    elif (
        method == "cumulative" or method == "two_parts"
    ):  # Accept both names for backward compatibility
        return cumulative_patching(
            source_model=source_model,
            target_model=target_model,
            split_layer=split_layer,
            in_place_patching=in_place_patching,
            reverse_cumulative_layers=reverse_cumulative_layers,
        )
    elif method == "layers_list":
        if layers is None:
            raise ValueError("layers parameter is required for 'layers_list' method")
        return layers_list_patching(
            source_model=source_model,
            target_model=target_model,
            layers=layers,
            in_place_patching=in_place_patching,
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'sliding_window', 'cumulative' or 'layers_list'"
        )
