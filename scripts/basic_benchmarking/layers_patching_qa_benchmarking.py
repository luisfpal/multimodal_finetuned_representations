from utils import (
    get_dataloader,
    benchmark_model_vqa_processed_dataloader,
)
from src.patching_benchmarking_common import (
    run_patching_benchmarking,
    create_common_parser,
    setup_patching_methods,
)

COCOQA_DATASET_ARGS = {
    "guide_text": "Answer the question using a single word or phrase.\n",
    "downsample_size": 2500,
    "answer_letters_with_processed_batch": True,
}
DATASETS = {
    "cocoqa_txt": {
        "texts_qa": True,
        **COCOQA_DATASET_ARGS,
    },
    "cocoqa_img": {
        "images_qa": True,
        **COCOQA_DATASET_ARGS,
    },
}


def main():
    """
    Benchmark model performance with layer patching between multimodal models.

    This script evaluates how patching layers from a source multimodal model
    to a target model (with pretrained connector) affects VQA performance.

    Two patching methods are supported:
    - sliding_window: Patch consecutive layers in a sliding window pattern
    - cumulative: Split model at a layer and patch the first part cumulatively

    Results are saved as CSV files in the following structure:
    results/layers_patching_qa_benchmarking/{dataset_name}/
    ├── {models}_sliding_window_ws{window_size}_s{stride}.csv
    └── {models}_cumulative_s{stride}.csv

    Each CSV contains accuracy metrics for different patching configurations
    plus baseline results for both the original multimodal model and the model
    with pretrained connector.

    When --save_baseline_outputs is used, baseline model answers are saved as JSON files:
    results/layers_patching_qa_benchmarking/{dataset_name}/
    ├── {models}_mm_model_outputs.json
    └── {models}_mm_pretrained_connector_outputs.json
    """

    parser = create_common_parser()
    args = parser.parse_args()

    # Setup patching methods configuration
    setup_patching_methods(args)

    run_patching_benchmarking(
        args=args,
        dataset_config=DATASETS,
        get_dataloader_func=get_dataloader,
        benchmark_func=benchmark_model_vqa_processed_dataloader,
        results_subdir="layers_patching_qa_benchmarking_v2",
    )


if __name__ == "__main__":
    main()
