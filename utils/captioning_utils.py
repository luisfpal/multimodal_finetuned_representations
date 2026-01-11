import torch
import json
from .cider import Cider
from typing import Dict, Optional, List, Tuple
import gc
from torch.utils.data import DataLoader, TensorDataset
from transformers.feature_extraction_utils import BatchFeature
from tqdm import tqdm

# from bert_score import score
from PIL import Image
from .model_utils import (
    ModelType,
    ProcessorType,
    load_hf_model_and_processor_or_tokenizer,
)


def generate_captions(
    model: ModelType,
    processor: ProcessorType,
    tokenized_batch: BatchFeature,
    max_new_tokens: int = 100,
) -> List[str]:
    """Generate captions for a batch of images."""
    _, input_ids_length = tokenized_batch.input_ids.shape[:2]
    with torch.no_grad():
        kwargs_for_generate = {
            **tokenized_batch.to(model.device),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        output_ids_tensor = model.generate(**kwargs_for_generate)

        # Extract only the generated tokens (excluding the prompt)
        generated_ids_only_tensor = output_ids_tensor[:, input_ids_length:]
        decoded_outputs_list = processor.batch_decode(
            generated_ids_only_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Clean up the generated captions
        predicted_captions = [
            decoded_output.strip() for decoded_output in decoded_outputs_list
        ]
    return predicted_captions


def save_captions_to_file(captions_data: List[Dict], file_path: str) -> None:
    """Save captions data to a JSON file with error handling."""
    try:
        with open(file_path, "w") as f:
            json.dump(captions_data, f, indent=2)
        print(f"Captions saved to: {file_path}")
    except Exception as e:
        print(f"Warning: Failed to save captions to {file_path}: {e}")


# def compute_bert_scores(generated_captions, list_of_reference_lists):
#     """
#     generated_captions: list of generated strings
#     list_of_reference_lists: list of list of reference captions
#                              e.g. [[ref1a, ref1b], [ref2a, ref2b], ...]
#     """
#     P, R, F1 = score(
#         cands=generated_captions,
#         refs=list_of_reference_lists,
#         lang="en",
#         verbose=False,
#     )
#     mean_F1 = F1.mean().item()
#     return mean_F1


def _get_generated_captions_from_list_of_dicts(list_of_dicts: List[Dict]) -> List[str]:
    """
    list_of_dicts: list of dicts with "caption" key
    """
    generated_captions = {}
    for d in list_of_dicts:
        if d["image_id"] not in generated_captions:
            generated_captions[d["image_id"]] = d["caption"]
    generated_captions = dict(sorted(generated_captions.items()))
    return list(generated_captions.values())


def _get_reference_captions_from_list_of_dicts(
    list_of_dicts: List[Dict],
) -> List[List[str]]:
    """
    list_of_dicts: list of dicts with "caption" key
    """
    reference_captions = {}
    for d in list_of_dicts:
        if d["image_id"] not in reference_captions:
            reference_captions[d["image_id"]] = []
        reference_captions[d["image_id"]].append(d["caption"])
    reference_captions = dict(sorted(reference_captions.items()))
    return list(reference_captions.values())


def _prepare_clip_inputs(
    images: List[Image.Image],
    captions: List[str],
    clip_processor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare CLIP model inputs from images and captions.

    Args:
        images: List of PIL images
        captions: List of caption strings
        clip_processor: CLIP processor for tokenization and image processing
        device: Target device for tensors

    Returns:
        Tuple of (pixel_values, input_ids, attention_mask)
    """
    image_inputs = clip_processor.image_processor(images, return_tensors="pt")
    text_inputs = clip_processor.tokenizer(
        captions, return_tensors="pt", padding=True, truncation=True
    )

    pixel_values = image_inputs.pixel_values.to(device)
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    return pixel_values, input_ids, attention_mask


def _compute_cosine_similarity_scores(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    weight: float = 2.5,
) -> torch.Tensor:
    """
    Compute weighted cosine similarity scores between image and text embeddings.

    Args:
        image_embeds: Image embeddings tensor
        text_embeds: Text embeddings tensor
        weight: Scaling factor for the scores

    Returns:
        Weighted cosine similarity scores
    """
    cosine_similarities = torch.nn.functional.cosine_similarity(
        image_embeds, text_embeds, dim=1
    )
    weighted_scores = weight * torch.clamp(cosine_similarities, min=0)
    return weighted_scores


def _compute_max_reference_similarity(
    generated_text_embeds: torch.Tensor,
    reference_captions: List[List[str]],
    clip_processor,
    clip_model,
) -> torch.Tensor:
    """
    Compute maximum cosine similarity between generated text embeddings and reference captions.

    Args:
        generated_text_embeds: Generated text embeddings tensor (batch_size, embed_dim)
        reference_captions: List of reference caption lists for each sample
        clip_processor: CLIP processor
        clip_model: CLIP model

    Returns:
        Maximum cosine similarities for each sample
    """
    device = clip_model.device
    image_size = clip_model.vision_model.config.image_size
    dummy_pixel_values = torch.zeros(1, 3, image_size, image_size, device=device)

    num_samples = len(generated_text_embeds)
    max_similarities = torch.zeros(num_samples, dtype=torch.float32, device=device)

    for sample_idx in range(num_samples):
        sample_generated_embed = generated_text_embeds[sample_idx]
        sample_reference_captions = reference_captions[sample_idx]

        # Tokenize reference captions
        tokenized_refs = clip_processor.tokenizer(
            sample_reference_captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = tokenized_refs.input_ids.to(device)
        attention_mask = tokenized_refs.attention_mask.to(device)

        # Get reference text embeddings
        with torch.no_grad():
            outputs = clip_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=dummy_pixel_values,
            )
            reference_text_embeds = outputs.text_embeds

        # Compute similarities and find maximum
        similarities = torch.nn.functional.cosine_similarity(
            sample_generated_embed.unsqueeze(0), reference_text_embeds, dim=1
        )
        max_similarities[sample_idx] = torch.clamp(similarities, min=0).max()

    return max_similarities


def harmonic_mean(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise harmonic mean of two tensors.

    Args:
        tensor1: First tensor
        tensor2: Second tensor

    Returns:
        Harmonic mean tensor

    Raises:
        ValueError: If tensors have different shapes or are not 1D
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same shape.")
    if tensor1.dim() != 1 or tensor2.dim() != 1:
        raise ValueError("Input tensors must be 1-dimensional.")

    numerator = 2 * tensor1 * tensor2
    denominator = tensor1 + tensor2

    # Avoid division by zero
    harmonic_means = torch.where(
        denominator != 0,
        numerator / denominator,
        torch.zeros_like(denominator),
    )

    return harmonic_means


def compute_clip_score(
    images: List[Image.Image],
    generated_captions: List[str],
    clip_processor,
    clip_model,
    weight: float = 2.5,
    batch_size: int = 16,
) -> float:
    """
    Compute CLIP score between images and generated captions.

    Args:
        images: List of PIL images
        generated_captions: List of generated caption strings (one per image)
        clip_processor: CLIP processor
        clip_model: CLIP model
        weight: Scaling factor for the scores
        batch_size: Batch size for processing

    Returns:
        Mean CLIP score across all samples
    """
    device = clip_model.device
    total_samples = len(images)
    all_scores = torch.zeros(total_samples, dtype=torch.float32, device=device)

    # Prepare inputs
    pixel_values, input_ids, attention_mask = _prepare_clip_inputs(
        images, generated_captions, clip_processor, device
    )

    # Create data loader for batch processing
    dataset = TensorDataset(pixel_values, input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    processed_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_pixel_values, batch_input_ids, batch_attention_mask = batch

            # Get CLIP embeddings
            outputs = clip_model(
                pixel_values=batch_pixel_values,
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )

            # Compute scores for current batch
            batch_scores = _compute_cosine_similarity_scores(
                outputs.image_embeds, outputs.text_embeds, weight
            )

            # Store scores
            current_batch_size = batch_input_ids.shape[0]
            start_idx = processed_samples
            end_idx = processed_samples + current_batch_size
            all_scores[start_idx:end_idx] = batch_scores
            processed_samples += current_batch_size

    return torch.mean(all_scores).item()


def compute_ref_clip_score(
    images: List[Image.Image],
    generated_captions: List[str],
    reference_captions: List[List[str]],
    clip_processor,
    clip_model,
    weight: float = 2.5,
    batch_size: int = 16,
) -> float:
    """
    Compute reference-aware CLIP score using harmonic mean of direct similarity
    and maximum similarity with reference captions.

    Args:
        images: List of PIL images
        generated_captions: List of generated caption strings (one per image)
        reference_captions: List of reference caption lists (one list per image)
        clip_processor: CLIP processor
        clip_model: CLIP model
        weight: Scaling factor for the scores
        batch_size: Batch size for processing

    Returns:
        Mean reference-aware CLIP score across all samples
    """
    device = clip_model.device
    total_samples = len(images)
    all_scores = torch.zeros(total_samples, dtype=torch.float32, device=device)

    # Prepare inputs
    pixel_values, input_ids, attention_mask = _prepare_clip_inputs(
        images, generated_captions, clip_processor, device
    )

    # Create data loader for batch processing
    dataset = TensorDataset(pixel_values, input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    processed_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_pixel_values, batch_input_ids, batch_attention_mask = batch

            # Get CLIP embeddings
            outputs = clip_model(
                pixel_values=batch_pixel_values,
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )

            current_batch_size = batch_input_ids.shape[0]
            start_idx = processed_samples
            end_idx = processed_samples + current_batch_size

            # Get reference captions for current batch
            batch_reference_captions = reference_captions[start_idx:end_idx]

            # Compute direct similarity scores
            direct_scores = _compute_cosine_similarity_scores(
                outputs.image_embeds, outputs.text_embeds, weight
            )

            # Compute maximum similarity with reference captions
            max_ref_similarities = _compute_max_reference_similarity(
                outputs.text_embeds,
                batch_reference_captions,
                clip_processor,
                clip_model,
            )

            # Compute harmonic mean of direct and reference similarities
            harmonic_scores = harmonic_mean(direct_scores, max_ref_similarities)

            # Store scores
            all_scores[start_idx:end_idx] = harmonic_scores
            processed_samples += current_batch_size

    return torch.mean(all_scores).item()


def compute_pycoco_scores(
    ground_truth_images_ids_pycoco,
    ground_truth_annotations_pycoco,
    generated_captions_pycoco,
):
    import os

    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["PATH"] = (
        f"/usr/lib/jvm/java-8-openjdk-amd64/bin:{os.environ.get('PATH', '')}"
    )

    import tempfile
    import contextlib
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO

    # Prepare COCO evaluation data
    ground_truth_coco_format = {
        "info": {},
        "licenses": [],
        "type": "captions",
        "images": ground_truth_images_ids_pycoco,
        "annotations": ground_truth_annotations_pycoco,
    }

    # Create temporary files for COCO evaluation
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_gt:
        json.dump(ground_truth_coco_format, f_gt)
        ground_truth_path = f_gt.name

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_res:
        json.dump(generated_captions_pycoco, f_res)
        results_path = f_res.name

    # Evaluate captions using COCO metrics
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        coco = COCO(ground_truth_path)
        coco_result = coco.loadRes(results_path)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params["image_id"] = coco_result.getImgIds()
        coco_eval.evaluate()

    return {
        "BLEU-1": coco_eval.eval["Bleu_1"],
        "BLEU-2": coco_eval.eval["Bleu_2"],
        "BLEU-3": coco_eval.eval["Bleu_3"],
        "BLEU-4": coco_eval.eval["Bleu_4"],
        "METEOR": coco_eval.eval["METEOR"],
        "ROUGE-L": coco_eval.eval["ROUGE_L"],
        "CIDEr": coco_eval.eval["CIDEr"],
        "SPICE": coco_eval.eval["SPICE"],
    }


def benchmark_model_captioning_processed_dataloader(
    model: ModelType,
    processor: ProcessorType,
    processed_dataloader: DataLoader,
    max_new_tokens: int = 50,
    save_captions_path: Optional[str] = None,
    clip_model_name_or_path: str = "openai/clip-vit-large-patch14-336",
    clip_cache_dir: str = "~/scratch/huggingface/hub",
    clip_weight: float = 2.5,
    clip_batch_size: int = 16,
) -> Dict[str, float]:
    """
    Benchmark model performance on captioning task with optional output saving.

    Args:
        model: The model to benchmark
        processor: The processor for the model
        processed_dataloader: DataLoader with processed data
        max_new_tokens: Maximum number of tokens to generate
        save_captions_path: Optional path to save generated captions
        clip_model_name_or_path: CLIP model name or path for evaluation
        clip_cache_dir: Cache directory for CLIP model
        clip_weight: Weight for CLIP score computation
        clip_batch_size: Batch size for CLIP evaluation

    Returns:
        Dictionary containing evaluation metrics including
        CIDEr, CLIP-S, RefCLIP-S scores
    """

    print("\n+-+-+-🔤 Captioning dataset+-+-+-+\n")
    dataset_size = len(processed_dataloader.dataset)
    progress_bar = tqdm(total=dataset_size, desc="Processing samples", unit="sample")
    update_every = max(1, int(0.1 * dataset_size))
    last_update = 0

    # Initialize data structures for evaluation
    generated_captions_pycoco = [None] * dataset_size
    ground_truth_images_ids_pycoco = [None] * dataset_size
    ground_truth_images = [None] * dataset_size
    ground_truth_annotations_pycoco = []
    annotation_id = 0
    sample_idx = 0

    # Process each batch
    for (
        tokenized_batch,
        reference_captions_batch,
        image_ids_batch,
        images_batch,
    ) in processed_dataloader:
        predicted_captions = generate_captions(
            model, processor, tokenized_batch, max_new_tokens
        )
        current_batch_size = len(predicted_captions)
        start_idx = sample_idx
        end_idx = sample_idx + current_batch_size
        ground_truth_images[start_idx:end_idx] = images_batch

        for batch_sample_idx in range(current_batch_size):
            # Use real image ID if available and valid, otherwise use sample index
            raw_image_id = (
                image_ids_batch[batch_sample_idx]
                if batch_sample_idx < len(image_ids_batch)
                else None
            )
            real_image_id = raw_image_id if raw_image_id is not None else sample_idx

            # Store generated caption for evaluation
            generated_captions_pycoco[sample_idx] = {
                "image_id": real_image_id,
                "caption": predicted_captions[batch_sample_idx],
            }

            # Add image to ground truth data
            ground_truth_images_ids_pycoco[sample_idx] = {"id": real_image_id}

            # Store all reference captions for this image
            for reference_caption in reference_captions_batch[batch_sample_idx]:
                ground_truth_annotations_pycoco.append(
                    {
                        "image_id": real_image_id,
                        "id": annotation_id,
                        "caption": reference_caption,
                    }
                )
                annotation_id += 1

            sample_idx += 1

        # Update progress bar
        if sample_idx % update_every == 0 or sample_idx == dataset_size:
            progress_bar.update(sample_idx - last_update)
            last_update = sample_idx

    progress_bar.close()
    torch.cuda.empty_cache()
    gc.collect()

    # Save captions if requested
    if save_captions_path:
        save_captions_to_file(generated_captions_pycoco, save_captions_path)

    # Get generated captions list, list of list of reference captions
    generated_captions = _get_generated_captions_from_list_of_dicts(
        generated_captions_pycoco
    )
    reference_captions = _get_reference_captions_from_list_of_dicts(
        ground_truth_annotations_pycoco
    )

    # Move model to CPU
    model_device = model.device
    model.to("cpu")

    # Load CLIP model for evaluation
    print("\n🖼️  Loading CLIP model for evaluation...")
    clip_model, clip_processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=clip_model_name_or_path,
        cache_dir=clip_cache_dir,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    clip_model.to(model_device)

    print("\n+-+-+-📝 Evaluating captions+-+-+-+\n")

    # Compute CLIP and reference-aware CLIP scores
    print("📊 Computing CLIP scores...")
    clip_score = compute_clip_score(
        ground_truth_images,
        generated_captions,
        clip_processor,
        clip_model,
        weight=clip_weight,
        batch_size=clip_batch_size,
    )
    # ref_clip_score = compute_ref_clip_score(
    #     ground_truth_images,
    #     generated_captions,
    #     reference_captions,
    #     clip_processor,
    #     clip_model,
    #     weight=clip_weight,
    #     batch_size=clip_batch_size,
    # )

    # Move model back to CPU
    clip_model.to("cpu")

    # Clean up
    torch.cuda.empty_cache()
    gc.collect()

    # Move model back to original device
    model.to(model_device)

    # Compute CIDEr score
    print("📊 Computing CIDEr score...")
    cider_format_predictions = {
        idx: [generated_captions[idx]] for idx in range(len(generated_captions))
    }
    cider_format_ground_truths = {
        idx: reference_captions[idx] for idx in range(len(reference_captions))
    }
    cider, _ = Cider(backend="pycocoeval").compute_score(
        cider_format_ground_truths, cider_format_predictions
    )

    return {
        "CIDEr": cider,
        "CLIP-S": clip_score,
        # "RefCLIP-S": ref_clip_score,
    }
