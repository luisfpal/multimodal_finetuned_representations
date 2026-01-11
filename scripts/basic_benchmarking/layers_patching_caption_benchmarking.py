from utils import (
    get_dataloader_for_captioning,
    benchmark_model_captioning_processed_dataloader,
)
from src.patching_benchmarking_common import (
    run_patching_benchmarking,
    create_common_parser,
    setup_patching_methods,
)

# Single dataset configuration for captioning
DATASETS = {
    "coco_captioning": {
        "downsample_size": 2500,
        "return_captions": True,
        "return_image_ids": True,
        "return_images": True,
    }
}

# CLIP and BERT score configuration
CLIP_CONFIG = {
    "clip_model_name_or_path": "openai/clip-vit-large-patch14-336",
    "clip_cache_dir": "~/scratch/huggingface/hub",
    "clip_weight": 2.5,
    "clip_batch_size": 16,
}


def main():
    """
    Benchmark model performance with layer patching between multimodal models.

    This script evaluates how patching layers from a source multimodal model
    to a target model (with pretrained connector) affects image captioning performance.

    Two patching methods are supported:
    - sliding_window: Patch consecutive layers in a sliding window pattern
    - cumulative: Split model at a layer and patch the first part cumulatively

    Results are saved as CSV files in the following structure:
    results/layers_patching_caption_benchmarking/{dataset_name}/
    ├── {models}_sliding_window_ws{window_size}_s{stride}.csv
    └── {models}_cumulative_s{stride}.csv

    Each CSV contains captioning metrics (BLEU, METEOR, ROUGE-L, CIDEr, SPICE, CLIP-S, RefCLIP-S, BERT-S)
    for different patching configurations plus baseline results for both
    the original multimodal model and the model with pretrained connector.

    When --save_baseline_outputs is used, baseline model captions are saved as JSON files:
    results/layers_patching_caption_benchmarking/{dataset_name}/
    ├── {multimodal_model_name}_outputs.json
    └── {multimodal_model_name}_{language_model_name}_pretrained_connector_outputs.json
    """

    parser = create_common_parser()
    args = parser.parse_args()

    # Setup patching methods configuration
    setup_patching_methods(args)

    run_patching_benchmarking(
        args=args,
        dataset_config=DATASETS,
        get_dataloader_func=get_dataloader_for_captioning,
        benchmark_func=benchmark_model_captioning_processed_dataloader,
        results_subdir="layers_patching_caption_benchmarking_v2",
    )


if __name__ == "__main__":
    main()
