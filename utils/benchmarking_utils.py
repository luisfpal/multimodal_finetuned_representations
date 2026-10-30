"""Utilities for benchmarking VQA models, including digit-to-word text normalization, answer parsing against ground truth, and saving answers to JSON."""

import torch
import gc
from typing import Dict, Optional, List
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model_utils import ModelType, ProcessorType
from .constants import COCOQA_VI_DIGITS_MAP
import json


def process_digits(text: str) -> str:
    """
    Replace digit strings in the text with their word equivalents.
    Handles ambiguity by processing longer digits first.
    """
    # Sort keys in descending order of length to prevent substring replacement issues
    for key in sorted(COCOQA_VI_DIGITS_MAP.keys(), key=len, reverse=True):
        text = text.replace(key, COCOQA_VI_DIGITS_MAP[key])
    return text


def parse_predicted_answer(predicted_text: str, ground_truth_answer: str) -> str:
    """
    Parse the model's predicted answer and check for a match against the ground truth.

    This function handles two types of evaluation:
    - Multiple-choice (single character answers, e.g., "A")
    - Open-ended (free-form text answers, e.g., "apple")

    It performs case-insensitive and trimmed matching, and handles partial matches for open-ended answers.

    Args:
        predicted_text (str): The raw output from the model.
        ground_truth_answer (str): The correct answer to compare against.

    Returns:
        str: The parsed answer if it matches expectations, otherwise 'FAILED'.
    """
    predicted_text = predicted_text.strip().lower()
    ground_truth_answer = ground_truth_answer.strip().lower()

    # Multiple-choice: expect exact match or contained match (e.g., "Answer: A")
    if len(ground_truth_answer) == 1:
        if predicted_text == ground_truth_answer:
            return ground_truth_answer
        if ground_truth_answer in predicted_text:
            return ground_truth_answer
        return "FAILED"

    # Open-ended: allow substring match (e.g., answer="apple", predicted="a green apple")
    predicted_text = process_digits(predicted_text)
    if ground_truth_answer in predicted_text or predicted_text in ground_truth_answer:
        return ground_truth_answer
    return "FAILED"


def save_answers_to_file(answers_data: List[Dict], file_path: str) -> None:
    """Save answers data to a JSON file with error handling."""
    try:
        with open(file_path, "w") as f:
            json.dump(answers_data, f, indent=2)
        print(f"Answers saved to: {file_path}")
    except Exception as e:
        print(f"Warning: Failed to save answers to {file_path}: {e}")


def benchmark_model_vqa_processed_dataloader(
    model: ModelType,
    processed_dataloader: DataLoader,
    processor: ProcessorType,
    max_new_tokens: int = 1,
    save_answers_path: Optional[str] = None,
) -> Dict[str, float]:
    """Benchmark model performance on VQA task with optional output saving."""
    correct_answers = 0
    incorrect_answers = 0

    dataset_size = len(processed_dataloader.dataset)
    progress_bar = tqdm(total=dataset_size, desc="Processing samples", unit="sample")
    update_every = max(1, int(0.1 * dataset_size))
    last_update = 0
    processed_samples = 0

    # Initialize answers for saving if requested
    answers_to_save = []
    if save_answers_path:
        answers_to_save = [None] * dataset_size

    # Process each batch
    for batch, ground_truth_answers in processed_dataloader:
        current_batch_size, input_ids_length = batch.input_ids.shape[:2]
        processed_samples += current_batch_size

        # Generate predictions
        with torch.no_grad():
            kwargs_for_generate = {
                **batch.to(model.device),
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

        # Clean up the predicted answers
        predicted_answers = [
            decoded_output.strip() for decoded_output in decoded_outputs_list
        ]

        # Evaluate each prediction
        for idx, (predicted_text, ground_truth_answer) in enumerate(
            zip(predicted_answers, ground_truth_answers)
        ):
            parsed_answer = parse_predicted_answer(predicted_text, ground_truth_answer)

            if parsed_answer == ground_truth_answer.lower():
                correct_answers += 1
            else:
                incorrect_answers += 1

            # Store answer for saving if requested
            if save_answers_path:
                sample_idx = processed_samples - current_batch_size + idx
                answers_to_save[sample_idx] = {
                    "sample_id": sample_idx,
                    "predicted_answer": predicted_text,
                    "parsed_answer": parsed_answer,
                    "ground_truth": ground_truth_answer,
                    "is_correct": parsed_answer == ground_truth_answer,
                }

        # Update progress bar
        if processed_samples % update_every == 0 or processed_samples == dataset_size:
            progress_bar.update(processed_samples - last_update)
            last_update = processed_samples

    progress_bar.close()
    torch.cuda.empty_cache()
    gc.collect()

    # Save answers if requested
    if save_answers_path:
        save_answers_to_file(answers_to_save, save_answers_path)

    total_answers = correct_answers + incorrect_answers
    accuracy = correct_answers / total_answers if total_answers > 0 else 0

    return {
        "num_correct_answers": correct_answers,
        "num_incorrect_answers": incorrect_answers,
        "accuracy": accuracy,
    }
