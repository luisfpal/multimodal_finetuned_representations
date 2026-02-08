from .benchmarking_utils import (
    benchmark_model_vqa_processed_dataloader,
    parse_predicted_answer,
)
from .captioning_utils import benchmark_model_captioning_processed_dataloader
from .constants import (
    ANSWER_TEXT,
    ASSISTANT_ROLE,
    COCOQA_VI_DIGITS_MAP,
    GUIDE_TEXT,
    LLAVA_CHAT_TEMPLATE,
    MMLU_ANSWER_CHOICES,
    SQA_ANSWER_CHOICES,
    LLaVA1_5_SYSTEM_INSTRUCTION,
    INTERNVL2_SYSTEM_INSTRUCTION,
    INTERNVL3_5_SYSTEM_INSTRUCTION,
    SYSTEM_ROLE,
)
from .dataset import (
    format_prompts,
    get_dataloader,
    get_dataloader_for_captioning,
)
from .layers_weights_patching import patch_layers_weights
from .metrics_utils import (
    compute_cross_layers_residual_stream_similarities,
    compute_heads_projection_residual_stream_similarities,
    compute_layers_cosine_similarity,
    compute_layers_homogeneity_score,
    compute_layers_intrinsic_dimension,
    compute_layers_residual_stream_entropy,
    compute_layers_residual_stream_similarities,
)
from .model_utils import (
    ModelType,
    ProcessorType,
    load_hf_model_and_processor_or_tokenizer,
    replace_multimodal_lm,
    replace_multimodal_projector,
)
from .multimodal_extraction_setup import setup_multimodal_model
from .operations_utils import seed_all
from .plot_utils import plot_similarity_measure_matrix
from .sample_img_txt_embedding_positions import sample_image_and_text_positions

__all__ = [
    "LLAVA_CHAT_TEMPLATE",
    "GUIDE_TEXT",
    "ANSWER_TEXT",
    "SYSTEM_ROLE",
    "LLaVA1_5_SYSTEM_INSTRUCTION",
    "INTERNVL2_SYSTEM_INSTRUCTION",
    "INTERNVL3_5_SYSTEM_INSTRUCTION",
    "ASSISTANT_ROLE",
    "SQA_ANSWER_CHOICES",
    "MMLU_ANSWER_CHOICES",
    "COCOQA_VI_DIGITS_MAP",
    "get_dataloader",
    "get_dataloader_for_captioning",
    "format_prompts",
    "compute_layers_residual_stream_entropy",
    "compute_layers_residual_stream_similarities",
    "compute_cross_layers_residual_stream_similarities",
    "compute_heads_projection_residual_stream_similarities",
    "compute_layers_intrinsic_dimension",
    "compute_layers_cosine_similarity",
    "compute_layers_homogeneity_score",
    "load_hf_model_and_processor_or_tokenizer",
    "replace_multimodal_lm",
    "replace_multimodal_projector",
    "ProcessorType",
    "ModelType",
    "seed_all",
    "setup_multimodal_model",
    "parse_predicted_answer",
    "benchmark_model_vqa_processed_dataloader",
    "patch_layers_weights",
    "sample_image_and_text_positions",
    "benchmark_model_captioning_processed_dataloader",
    "plot_similarity_measure_matrix",
]
