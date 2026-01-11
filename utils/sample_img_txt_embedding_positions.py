import torch

def sample_image_and_text_positions(
    input_ids: torch.Tensor,
    image_token_id: int = 32000,
    image_seq_length: int = 576,
    pad_token_id: int = 32001,
    text_direction: str = "right",
    skip_image_pos: bool = False,
    skip_text_pos: bool = False,
    generator: torch.Generator = None,
    subtract_padding_per_sample: bool = True,
    chat_mode: bool = False,
) -> torch.Tensor:
    """
    Sample image and text positions (indices) from input_ids tensor.
    
    Args:
        input_ids: Input token IDs tensor of shape (batch_size, sequence_length)
        image_token_id: Token ID that marks the start of image tokens
        image_seq_length: Length of image token sequence
        pad_token_id: Token ID for padding tokens
        text_direction: Direction to sample text tokens from ("left" or "right" of image)
        skip_image_pos: Whether to skip sampling image positions
        skip_text_pos: Whether to skip sampling text positions
        generator: Random generator for reproducible sampling (will be moved to input_ids device if provided)
        subtract_padding_per_sample: If True, subtract padding positions per sample to avoid bias.
                                    This is useful for the residual stream tracer.
                                   If False, use original positions (batch-wide consideration).
                                   Not necessary if not working with the residual stream tracer.
        chat_mode: If True, the inputs contain the ASSISTANT token embeddings at the end of the sequence.
                    Thus, they are removed from the sampling process.
    Returns:
        Tensor of sampled positions:
        - If both image and text: shape (batch_size, 2) with [image_pos, text_pos]
        - If only image: shape (batch_size,) with image positions
        - If only text: shape (batch_size,) with text positions
        
    Raises:
        ValueError: If invalid parameters or no valid positions found
    """
    if text_direction not in {"left", "right"}:
        raise ValueError("text_direction must be 'left' or 'right'")
    if skip_image_pos and skip_text_pos:
        raise ValueError("Cannot skip both image and text positions")

    device = input_ids.device
    
    # Remove the assistant tokens from the input_ids if they are present
    # when the input modalities are image-text
    if chat_mode and skip_image_pos:
        # ~7-10 tokens for: ASSISTANT: Answer:
        approximate_max_observed_assistant_tokens_length = 10
        input_ids = input_ids[:, :-approximate_max_observed_assistant_tokens_length]
    
    batch_size, seq_length = input_ids.shape

    # Create generator on the correct device if not provided
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(42)
        
    # Create padding mask
    padding_mask = (input_ids == pad_token_id)  # (batch_size, seq_length)
    
    # Calculate padding tokens lengths
    padding_tokens_lengths = padding_mask.sum(dim=1).to(device) # (batch_size,)
    
    # Find image start positions
    image_start_mask = (input_ids == image_token_id)
    if not image_start_mask.any():
        raise ValueError("No image tokens found in input_ids")
    
    # Get first occurrence of image token for each batch element
    image_start = image_start_mask.int().argmax(dim=1)  # (batch_size,)
    image_end = image_start + image_seq_length  # (batch_size,)
    
    # Validate image positions
    if (image_end > seq_length).any():
        sampled_image_end = image_end[image_end > seq_length][0]
        raise ValueError(f"Image sequence extends beyond sequence length: {sampled_image_end} > {seq_length}")

    # Sample image positions
    if not skip_image_pos:
        image_offsets = torch.randint(
            0, image_seq_length, size=(batch_size,), generator=generator, device=device
        )
        sampled_image_pos = image_start + image_offsets  # (batch_size,)
        
        # Adjust image positions by subtracting padding tokens lengths
        if subtract_padding_per_sample:
            sampled_image_pos = sampled_image_pos - padding_tokens_lengths

        if skip_text_pos:
            return sampled_image_pos

    # Sample text positions
    if not skip_text_pos:
        # Create position indices
        position_indices = torch.arange(seq_length, device=device).unsqueeze(0)  # (1, seq_length)
        
        # Create valid text mask (exclude padding tokens)
        valid_text_mask = input_ids != pad_token_id  # (batch_size, seq_length)
        
        # !specific to the experiments I am running
        # and modifies the image_end variable when it is no longer used for sampling the image positions
        if chat_mode:
            # ~38-40 tokens for:
            # f"Below are {len(captions)} numbered descriptions of the image.
            # Please generate an additional description of the image.
            # The caption should be a single sentence, not multiple sentences!!!\n\n"
            approximate_max_observed_captions_tokens_length = 40
            image_end += approximate_max_observed_captions_tokens_length

        # Create directional mask based on text_direction
        if text_direction == "right":
            directional_mask = position_indices >= image_end.unsqueeze(1)  # (batch_size, seq_length)
        elif text_direction == "left":
            directional_mask = position_indices < image_start.unsqueeze(1)  # (batch_size, seq_length)
        else:
            raise ValueError(f"Invalid text_direction: {text_direction}. Must be 'left' or 'right'.")

        # Combine masks to get candidate positions
        candidate_mask = valid_text_mask & directional_mask  # (batch_size, seq_length)

        # Check if any batch elements have valid text positions
        valid_positions_per_batch = candidate_mask.sum(dim=1)  # (batch_size,)
        if (valid_positions_per_batch == 0).any():
            raise ValueError(
                f"Some batch elements have no valid text tokens in the '{text_direction}' direction"
            )

        # Create sampling probabilities (uniform over valid positions)
        sampling_probs = candidate_mask.float()
        sampling_probs = sampling_probs / sampling_probs.sum(dim=1, keepdim=True)  # Normalize

        # Sample one text position per batch element
        sampled_text_pos = torch.multinomial(
            sampling_probs, num_samples=1, replacement=False, generator=generator
        ).squeeze(1)  # (batch_size,)
        
        # Adjust text positions by subtracting padding tokens lengths
        if subtract_padding_per_sample:
            sampled_text_pos = sampled_text_pos - padding_tokens_lengths

        if skip_image_pos:
            return sampled_text_pos

    # Return both image and text positions
    return torch.stack([sampled_image_pos, sampled_text_pos], dim=1)  # (batch_size, 2)