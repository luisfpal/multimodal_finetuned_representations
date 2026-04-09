"""Plot encoder-decoder similarity (CKA or OLS R²) between encoder and VLM decoder residual stream.

Produces a 1x3 subplot per combination of (encoder, dataset, pooling, caption_source, norm, measure, N):
  [Text Subset] | [Image Subset] | [Concat]

CKA: one line per model.
OLS R²: two lines per model — test (solid) and train (dashed), same color.

Results naming convention (from similarity_vlms_encoder_decoder.py):
  {decoder_name}_{encoder_key}_{pooling_method}_{caption_source}_{norm_tag}_{measure}_N{num_samples}.safetensors
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file

sys.path.append(str(Path.cwd().parent))

# ============================================================================
# CONFIGURATION — edit this section to control what gets plotted
# ============================================================================

DATASETS = [
    # "l11p/COCOQA-restval-ICQA",
    "l11p/VQAv2-COCO-restval-ICQ",
]

MODELS = {
    "LLaVA-7B Finetuned": "llava_7b",
    "LLaVA-7B Pretrained": "llava-v1.5-7b-pretrained-lbasile",
    # "LLaVA-13B Finetuned": "llava_13b",
    # "LLaVA-13B Pretrained": "llava-v1.5-13b-pretrained-lbasile",
    # "InternVL2-8B": "InternVL2-8B",
    # "InternVL2-8B-Pretrained": "InternVL2-8B-Pretrain",
    # "Cambrian-8B": "cambrian-8b",
    # "Cambrian-8B-Pretrained": "cambrian-8b-pretrained",
}

# ENCODER_MODELS = ["jina"]
ENCODER_MODELS = ["clip"]
# ENCODER_MODELS = ["jina", "clip"]

POOLING_METHODS = ["last"]
# POOLING_METHODS = ["last", "mean_context"]

CAPTION_SOURCES = ["human"]
# CAPTION_SOURCES = ["synthetic", "human"]

NORM_VARIANTS = ["unnormalized"]
# NORM_VARIANTS = ["unnormalized", "normalized"]

SIMILARITY_MEASURES = ["ols"]
# SIMILARITY_MEASURES = ["cka", "ols"]

N_SAMPLES_LIST = [3000]
# N_SAMPLES_LIST = [1000, 3000]

DATASET_SPLIT = "split01"

RESULTS_BASE = Path("../results/similarity_encoder_decoder")
PLOTS_DIR = Path("../plots/similarity_encoder_decoder")

# ============================================================================
# CONSTANTS
# ============================================================================

SUBSET_TITLES = ["Text Subset", "Image Subset", "Concat"]

CKA_KEYS = ["cka_txt", "cka_img", "cka_concat"]
OLS_TEST_KEYS = ["r2_test_txt", "r2_test_img", "r2_test_concat"]
OLS_TRAIN_KEYS = ["r2_train_txt", "r2_train_img", "r2_train_concat"]
OLS_KEYS = OLS_TEST_KEYS + OLS_TRAIN_KEYS


# ============================================================================
# HELPERS
# ============================================================================


def make_dataset_slug(dataset_name: str) -> str:
    slug = dataset_name.split("/")[-1].replace("-", "_").lower()
    return f"{slug}_{DATASET_SPLIT}"


def load_results(
    dataset_slug: str,
    model_id: str,
    encoder_key: str,
    pooling_method: str,
    caption_source: str,
    norm_tag: str,
    measure: str,
    num_samples: int,
) -> dict[str, np.ndarray] | None:
    filename = (
        f"{model_id}_{encoder_key}_{pooling_method}"
        f"_{caption_source}_{norm_tag}_{measure}_N{num_samples}.safetensors"
    )
    path = RESULTS_BASE / dataset_slug / filename
    keys = CKA_KEYS if measure == "cka" else OLS_KEYS
    try:
        data = load_file(str(path))
        return {key: data[key].cpu().numpy() for key in keys}
    except Exception as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


# ============================================================================
# PLOTTING
# ============================================================================


def plot_results(
    dataset_name: str,
    encoder_key: str,
    pooling_method: str,
    caption_source: str,
    norm_tag: str,
    measure: str,
    num_samples: int,
    plots_dir: Path,
) -> None:
    dataset_slug = make_dataset_slug(dataset_name)
    dataset_short = dataset_name.split("/")[-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    has_data = False
    prop_cycle = iter(plt.rcParams["axes.prop_cycle"])

    for label, model_id in MODELS.items():
        results = load_results(
            dataset_slug, model_id, encoder_key, pooling_method,
            caption_source, norm_tag, measure, num_samples,
        )
        if results is None:
            continue
        has_data = True

        color = next(prop_cycle)["color"]
        first_key = CKA_KEYS[0] if measure == "cka" else OLS_TEST_KEYS[0]
        layers = np.arange(len(results[first_key]))

        if measure == "cka":
            for ax, key, title in zip(axes, CKA_KEYS, SUBSET_TITLES):
                ax.plot(layers, results[key], marker="o", markersize=3,
                        color=color, label=label)
                ax.set_title(title, fontsize=13)
        else:  # ols
            for ax, test_key, train_key, title in zip(
                axes, OLS_TEST_KEYS, OLS_TRAIN_KEYS, SUBSET_TITLES
            ):
                ax.plot(layers, results[test_key], marker="o", markersize=3,
                        linestyle="-", color=color, label=f"{label} (test)")
                ax.plot(layers, results[train_key], marker="o", markersize=3,
                        linestyle="--", color=color, label=f"{label} (train)")
                ax.set_title(title, fontsize=13)

    if not has_data:
        plt.close(fig)
        print(
            f"  Skipping: no data for {dataset_short} | encoder={encoder_key} | "
            f"{pooling_method} | {caption_source} | {norm_tag} | {measure} | N={num_samples}"
        )
        return

    ylabel = "Linear CKA" if measure == "cka" else "R² Score"
    for ax in axes:
        ax.set_xlabel("Decoder Layer", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        if measure == "cka":
            ax.set_ylim(0, 1)
    axes[0].set_ylabel(ylabel, fontsize=11)

    fig.suptitle(
        f"Encoder-Decoder {measure.upper()}: {dataset_short} | encoder={encoder_key}\n"
        f"pooling={pooling_method} | captions={caption_source} | {norm_tag} | N={num_samples}",
        fontsize=13,
    )
    plt.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_name = (
        f"{measure}_{dataset_slug}_{encoder_key}_{pooling_method}"
        f"_{caption_source}_{norm_tag}_N{num_samples}.png"
    )
    fig.savefig(plots_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plots_dir / out_name}")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    for encoder_key in ENCODER_MODELS:
        for dataset_name in DATASETS:
            for pooling_method in POOLING_METHODS:
                for caption_source in CAPTION_SOURCES:
                    for norm_tag in NORM_VARIANTS:
                        for measure in SIMILARITY_MEASURES:
                            for num_samples in N_SAMPLES_LIST:
                                plot_results(
                                    dataset_name=dataset_name,
                                    encoder_key=encoder_key,
                                    pooling_method=pooling_method,
                                    caption_source=caption_source,
                                    norm_tag=norm_tag,
                                    measure=measure,
                                    num_samples=num_samples,
                                    plots_dir=PLOTS_DIR,
                                )

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
