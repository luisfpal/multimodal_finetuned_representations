"""
Configuration-based results loader for multimodal analysis.
Loads experimental results using YAML configuration for flexible data access.
"""

import torch
import yaml
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Any, Union, Optional
import pandas as pd


class ResultsLoader:
    """
    Load and organize experimental results using YAML configuration.

    Provides clean access to model measures, similarities, transplantation results,
    modalities similarities, and caption benchmarking results.

    The results data are organized in a dictionary with the following structure:
    - data["model_data_measures"]
        - dataset
            - model
                - dataset_entropy
                    - residual_stream_type_pooling_args e.g,
                        - output_layer_mean ...
                - prompt_entropy, e.g.,
                    - output_layer_mean ...
                - intrinsic_dimension, e.g.,
                    - output_layer_mean_k<k>_r<r>
    - data["models_similarities"]
        - dataset
            - residual_stream_type_pooling dictionary
                - measure_name, e.g., neighborhood_overlap, linear_cka  # svcca commented out
                    - similarity_matrix tensor of shape (num_heads, num_layers) or (num_layers,)
    - data["transplanting_layers_qa_benchmarking"]
        - dataset
            - experiment_name, e.g.,
                - sliding_window_ws<ws>_s<s>
                - two_parts_s<s>
    - data["modalities_similarities"]
        - model_name
            - dataset_name
                - stream_type
                    - layers_cosine_similarity tensor of shape (layers,)
                    - homogeneity_score_<metric> tensor of shape (layers,) for each metric (euclidean, cosine)
    - data["transplanting_layers_caption_benchmarking"]
        - dataset_name
            - experiment_name, e.g.,
                - sliding_window_ws<ws>_s<s>
                - two_parts_s<s>
                    - DataFrame with captioning metrics
    """

    def __init__(self, config_path: Union[str, Path] = None):
        """Initialize loader with configuration file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize results directories
        self.results_dir = config_path.parent / self.config["paths"]["results"]
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        # Initialize coco_captioning results directory if specified
        if "results_coco_captioning" in self.config["paths"]:
            self.results_coco_captioning_dir = (
                config_path.parent / self.config["paths"]["results_coco_captioning"]
            )
            if not self.results_coco_captioning_dir.exists():
                print(
                    f"⚠️  Coco captioning results directory not found: {self.results_coco_captioning_dir}"
                )
                self.results_coco_captioning_dir = None
        else:
            self.results_coco_captioning_dir = None

        self.data = {}

    def load_all(self) -> Dict[str, Any]:
        """Load all available results."""
        print("\n🚀 Loading results...")

        # Load all data types
        self.data["model_data_measures"] = self._load_model_data_measures()
        self.data["models_similarities"] = self._load_models_similarities()
        self.data["transplanting_layers_qa_benchmarking"] = (
            self._load_transplanting_layers_qa_benchmarking()
        )
        self.data["modalities_similarities"] = self._load_modalities_similarities()
        self.data["transplanting_layers_caption_benchmarking"] = (
            self._load_transplanting_layers_caption_benchmarking()
        )

        print(f"✅ Loaded: {list(self.data.keys())}")
        return self.data

    def _load_model_data_measures(self) -> Dict[str, Any]:
        """Load model measures (entropy, intrinsic dimension)."""
        model_data_measures = {}

        for dataset in self.config["datasets"]:
            model_data_measures[dataset] = {}

            # Try loading from main results directory
            models_dir = self.results_dir / dataset / "models_data_measures"
            if models_dir.exists():
                for model_dir in models_dir.iterdir():
                    if model_dir.is_dir():
                        model_key = self.config["models"].get(
                            model_dir.name, model_dir.name
                        )
                        model_data_measures[dataset][model_key] = (
                            self._load_model_measures(model_dir)
                        )

            # Try loading from coco_captioning results directory if available
            if self.results_coco_captioning_dir is not None:
                coco_models_dir = (
                    self.results_coco_captioning_dir / dataset / "models_data_measures"
                )
                if coco_models_dir.exists():
                    for model_dir in coco_models_dir.iterdir():
                        if model_dir.is_dir():
                            model_key = self.config["models"].get(
                                model_dir.name, model_dir.name
                            )
                            # Only load if not already loaded from main directory
                            if model_key not in model_data_measures[dataset]:
                                model_data_measures[dataset][model_key] = (
                                    self._load_model_measures(model_dir)
                                )

            if not model_data_measures[dataset]:
                print(f"⚠️  No model data measures found for {dataset}")

        return model_data_measures

    def _load_model_measures(self, model_dir: Path) -> Dict[str, Any]:
        """Load measures for a single model."""
        model_data = {
            "dataset_entropy": {},
            "prompt_entropy": {},
            "intrinsic_dimension": {},
        }

        for file_path in model_dir.glob("*.safetensors"):
            filename = file_path.stem

            try:
                data = load_file(str(file_path))
            except Exception as e:
                print(f"⚠️  Failed to load {filename}: {e}")
                continue

            if "plain" in filename:
                filename = filename.replace("_plain", "")
            elif "chat" in filename:
                filename = filename.replace("_chat", "")

            if filename.startswith("dataset_entropy_"):
                key = filename.replace("dataset_entropy_", "")
                model_data["dataset_entropy"][key] = data

            elif filename.startswith("prompt_entropy_"):
                key = filename.replace("prompt_entropy_", "")
                model_data["prompt_entropy"][key] = data

            elif filename.startswith("id_"):
                parts = filename.split("_")
                k, r = parts[1], parts[2]
                key = f"{filename.replace(f'id_{k}_{r}_', '')}_k{k}_r{r}"
                model_data["intrinsic_dimension"][key] = data

        return model_data

    def _load_models_similarities(self) -> Dict[str, Any]:
        """Load similarity measures between models."""
        models_similarities = {}

        for dataset in self.config["datasets"]:
            models_similarities[dataset] = {}

            # Try loading from main results directory
            sim_dir = self.results_dir / dataset / "similarities"
            if sim_dir.exists():
                for file_path in sim_dir.glob("*.safetensors"):
                    filename = file_path.stem

                    try:
                        data = load_file(str(file_path))
                        # dictionary with :
                        #  - neighborhood_overlap
                        #  - linear_cka
                        #  - svcca  # commented out
                    except Exception as e:
                        print(f"⚠️  Failed to load {filename}: {e}")
                        continue

                    if "_vs_" in filename:
                        # Extract stream_type and pooling from filename
                        parts = filename.split("_vs_")[-1].split("_")
                        stream_type = "_".join(parts[1:-1])
                        pooling = parts[-1]
                        key = f"{stream_type}_{pooling}"

                        # Initialize nested structure if needed
                        if key not in models_similarities[dataset]:
                            models_similarities[dataset][key] = {}

                        # The loaded data is already a dictionary with measures as keys
                        # Merge the data into our structure
                        for measure_name, measure_data in data.items():
                            models_similarities[dataset][key][measure_name] = (
                                measure_data
                            )

            # Try loading from coco_captioning results directory if available
            if self.results_coco_captioning_dir is not None:
                coco_sim_dir = (
                    self.results_coco_captioning_dir / dataset / "similarities"
                )
                if coco_sim_dir.exists():
                    for file_path in coco_sim_dir.glob("*.safetensors"):
                        filename = file_path.stem

                        try:
                            data = load_file(str(file_path))
                        except Exception as e:
                            print(f"⚠️  Failed to load {filename}: {e}")
                            continue

                        if "_vs_" in filename:
                            # Extract stream_type and pooling from filename
                            parts = filename.split("_vs_")[-1].split("_")
                            stream_type = "_".join(parts[1:-1])
                            pooling = parts[-1]
                            key = f"{stream_type}_{pooling}"

                            # Initialize nested structure if needed
                            if key not in models_similarities[dataset]:
                                models_similarities[dataset][key] = {}

                            # The loaded data is already a dictionary with measures as keys
                            # Merge the data into our structure
                            for measure_name, measure_data in data.items():
                                models_similarities[dataset][key][measure_name] = (
                                    measure_data
                                )

            if not models_similarities[dataset]:
                print(f"⚠️  No model similarities found for {dataset}")

        return models_similarities

    def create_similarities_difference_dataset(
        self,
        source_dataset1: str = "cocoqa_txt",
        source_dataset2: str = "cocoqa_img",
        target_dataset: str = "cocoqa_txt_minus_cocoqa_img",
        verbose: bool = False,
    ) -> bool:
        """
        Create a difference dataset by subtracting similarities.

        Args:
            source_dataset1: First dataset (e.g., "cocoqa_txt")
            source_dataset2: Second dataset to subtract (e.g., "cocoqa_img")
            target_dataset: Name for the new difference dataset
            verbose: Whether to print detailed progress

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "=" * 50)
        print("➕ Creating similarities difference dataset...")
        print("=" * 50)

        if "models_similarities" not in self.data:
            print("⚠️  No similarity data loaded")
            return False

        models_similarities = self.data["models_similarities"]

        # Check if both source datasets exist
        if (
            source_dataset1 not in models_similarities
            or source_dataset2 not in models_similarities
        ):
            print(
                f"⚠️  Cannot create difference dataset: missing {source_dataset1} or {source_dataset2}"
            )
            return False

        models_similarities[target_dataset] = {}

        # Get all common stream_type_pooling keys
        dataset1_stream_pooling_keys = set(models_similarities[source_dataset1].keys())
        dataset2_stream_pooling_keys = set(models_similarities[source_dataset2].keys())
        common_stream_pooling_keys = dataset1_stream_pooling_keys.intersection(
            dataset2_stream_pooling_keys
        )

        if not common_stream_pooling_keys:
            print(
                f"⚠️  No common stream_pooling_keys found between {source_dataset1} and {source_dataset2}"
            )
            return False

        success_count = 0
        for stream_pooling_key in common_stream_pooling_keys:
            models_similarities[target_dataset][stream_pooling_key] = {}

            # Get all common measure keys
            dataset1_measures = set(
                models_similarities[source_dataset1][stream_pooling_key].keys()
            )
            dataset2_measures = set(
                models_similarities[source_dataset2][stream_pooling_key].keys()
            )
            common_measures = dataset1_measures.intersection(dataset2_measures)

            for measure in common_measures:
                data1 = models_similarities[source_dataset1][stream_pooling_key][
                    measure
                ]
                data2 = models_similarities[source_dataset2][stream_pooling_key][
                    measure
                ]

                # Perform subtraction: dataset1 - dataset2
                try:
                    diff_data = data1 - data2
                    models_similarities[target_dataset][stream_pooling_key][measure] = (
                        diff_data
                    )
                    if verbose:
                        print(f"✅ Created difference: {stream_pooling_key}/{measure}")
                    success_count += 1
                except Exception as e:
                    print(
                        f"⚠️  Failed to compute difference for {stream_pooling_key}/{measure}: {e}"
                    )

        if success_count > 0:
            print(
                f"✅ Similarities difference dataset '{target_dataset}' created successfully with {success_count} measures"
            )
            print("=" * 50)
            return True
        else:
            print(
                f"❌ Failed to create similarities difference dataset '{target_dataset}'"
            )
            return False

    def compute_mean_heads_projections_at_layers(self, verbose: bool = False) -> bool:
        """
        Compute mean across heads (columns) for heads_projection similarity matrices.
        Creates new stream_type_pooling fields: mean_heads_projection_mean, mean_heads_projection_last

        For each heads_projection matrix with shape (layers, heads), computes column mean
        resulting in shape (layers,) and stores as new measure.

        Args:
            verbose: Whether to print detailed progress

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "=" * 50)
        print("➕ Computing mean heads projections at layers...")
        print("=" * 50)

        if "models_similarities" not in self.data:
            print("⚠️  No similarity data loaded")
            return False

        models_similarities = self.data["models_similarities"]
        success_count = 0

        # Process all datasets (including difference datasets)
        for dataset_name, dataset_data in models_similarities.items():
            if verbose:
                print(f"\n📊 Processing heads mean for dataset: {dataset_name}")

            # Look for heads_projection stream types
            heads_projection_keys = [
                key
                for key in dataset_data.keys()
                if key.startswith("heads_projection_")
                and key.endswith(("_mean", "_last"))
            ]

            if not heads_projection_keys:
                if verbose:
                    print(f"  ⚠️  No heads_projection data found for {dataset_name}")
                continue

            for heads_key in heads_projection_keys:
                # Extract pooling type (mean or last)
                pooling_type = heads_key.split("_")[-1]  # "mean" or "last"
                new_key = f"mean_heads_projection_{pooling_type}"

                # Initialize new stream_type_pooling if not exists
                if new_key not in dataset_data:
                    dataset_data[new_key] = {}

                measures_data = dataset_data[heads_key]

                for measure_name, measure_matrix in measures_data.items():
                    try:
                        # Compute mean across columns (heads dimension)
                        # (layers, heads) -> (layers,)
                        if hasattr(measure_matrix, "mean"):
                            heads_mean = measure_matrix.mean(dim=1)

                        dataset_data[new_key][measure_name] = heads_mean

                        if verbose:
                            print(
                                f"  ✅ {heads_key}/{measure_name} -> {new_key}/{measure_name}"
                            )
                            print(
                                f"     Shape: {measure_matrix.shape} -> {heads_mean.shape}"
                            )
                        success_count += 1

                    except Exception as e:
                        print(
                            f"  ⚠️  Failed to compute heads mean for {heads_key}/{measure_name}: {e}"
                        )

        if success_count > 0:
            print(
                f"\n✅ Mean heads projections at layers completed successfully: {success_count} measures processed"
            )
            print("=" * 50)
            return True
        else:
            print("\n❌ Failed to compute mean heads projections at layers")
            return False

    def create_dataset_measures_models_difference(
        self,
        dataset_name: str,
        model1: str,
        model2: str,
        target_model: str = None,
        verbose: bool = False,
    ) -> bool:
        """
        Create difference between model measures: model1 - model2 for a specific dataset.

        Computes differences for all matching measure keys across all measure types:
        - dataset_entropy
        - prompt_entropy
        - intrinsic_dimension

        Args:
            dataset_name: Dataset to process (e.g., "cocoqa_img")
            model1: First model (e.g., "base")
            model2: Second model to subtract (e.g., "pretrained")
            target_model: Name for difference model (default: "model1-model2")
            verbose: Whether to print detailed progress

        Returns:
            bool: True if successful, False otherwise

        Example:
            results.create_dataset_measures_models_difference("cocoqa_img", "base", "pretrained")
            # Creates: data["model_data_measures"]["cocoqa_img"]["base-pretrained"]
        """
        if target_model is None:
            target_model = f"{model1}-{model2}"

        print("\n" + "=" * 50)
        print(f"➕ Creating model measures difference: {model1} - {model2}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Target model: {target_model}")
        print("=" * 50)

        if "model_data_measures" not in self.data:
            print("⚠️  No model data measures loaded")
            return False

        model_data_measures = self.data["model_data_measures"]

        # Check if dataset exists
        if dataset_name not in model_data_measures:
            print(f"⚠️  Dataset '{dataset_name}' not found")
            print(f"   Available datasets: {list(model_data_measures.keys())}")
            return False

        dataset_data = model_data_measures[dataset_name]

        # Check if both models exist
        if model1 not in dataset_data or model2 not in dataset_data:
            print(
                f"⚠️  Cannot create difference: missing model '{model1}' or '{model2}'"
            )
            print(f"   Available models: {list(dataset_data.keys())}")
            return False

        # Check if both models have the same measures
        model1_measures = dataset_data[model1].keys()
        model2_measures = dataset_data[model2].keys()
        if model1_measures != model2_measures:
            print(f"⚠️  Models {model1} and {model2} have different measures")
            return False

        # Initialize target model structure
        dataset_data[target_model] = {
            measure_type: {} for measure_type in model1_measures
        }

        success_count = 0

        for measure_type in model1_measures:
            if verbose:
                print(f"\n📊 Processing {measure_type}...")

            model1_stream_pooling_measures = dataset_data[model1].get(measure_type, {})
            model2_stream_pooling_measures = dataset_data[model2].get(measure_type, {})

            # Get common residual_stream_type_pooling keys
            model1_stream_type_pooling_keys = set(model1_stream_pooling_measures.keys())
            model2_stream_type_pooling_keys = set(model2_stream_pooling_measures.keys())
            common_stream_type_pooling_keys = (
                model1_stream_type_pooling_keys.intersection(
                    model2_stream_type_pooling_keys
                )
            )

            if not common_stream_type_pooling_keys:
                if verbose:
                    print(f"  ⚠️  No common {measure_type} measures found")
                continue

            for stream_type_pooling_key in common_stream_type_pooling_keys:
                try:
                    data1 = next(
                        iter(
                            model1_stream_pooling_measures[
                                stream_type_pooling_key
                            ].values()
                        )
                    )
                    data2 = next(
                        iter(
                            model2_stream_pooling_measures[
                                stream_type_pooling_key
                            ].values()
                        )
                    )

                    # Perform subtraction: model1 - model2
                    diff_data = data1 - data2
                    dataset_data[target_model][measure_type][
                        stream_type_pooling_key
                    ] = diff_data

                    if verbose:
                        print(f"  ✅ {measure_type}/{stream_type_pooling_key}")
                        if hasattr(diff_data, "shape"):
                            print(f"     Shape: {diff_data.shape}")

                    success_count += 1

                except Exception as e:
                    print(
                        f"  ⚠️  Failed to compute difference for {measure_type}/{stream_type_pooling_key}: {e}"
                    )

        if success_count > 0:
            print(
                f"\n✅ Model measures difference '{target_model}' created successfully"
            )
            print(
                f"   Processed {success_count} measures across {len(model1_measures)} types"
            )

            if verbose:
                # Print summary of created measures
                print(f"\n📋 Created measures for '{target_model}':")
                for measure_type in dataset_data[target_model].keys():
                    created_measures = list(
                        dataset_data[target_model][measure_type].keys()
                    )
                    if created_measures:
                        print(f"   {measure_type}: {created_measures} measures")

            print("=" * 50)
            return True
        else:
            print(f"❌ Failed to create model measures difference '{target_model}'")
            return False

    def _load_transplanting_layers_qa_benchmarking(self) -> Dict[str, Any]:
        """Load transplantation experiment results."""
        transplanting_layers_qa_benchmarking = {}
        transplant_dir = self.results_dir / "transplanting_layers_qa_benchmarking"

        if not transplant_dir.exists():
            print("⚠️  No transplanting layers benchmarking results found")
            return transplanting_layers_qa_benchmarking

        for dataset in self.config["datasets"]:
            dataset_dir = transplant_dir / dataset
            if not dataset_dir.exists():
                continue

            transplanting_layers_qa_benchmarking[dataset] = {}
            for file_path in dataset_dir.glob("*.csv"):
                filename = file_path.stem

                try:
                    data = pd.read_csv(file_path)
                except Exception as e:
                    print(f"⚠️  Failed to load {filename}: {e}")
                    continue

                if "sliding_window" in filename:
                    key = filename.split("sliding_window_")[-1]
                    transplanting_layers_qa_benchmarking[dataset][
                        f"sliding_window_{key}"
                    ] = data
                elif "two_parts" in filename:
                    key = filename.split("two_parts_")[-1]
                    transplanting_layers_qa_benchmarking[dataset][
                        f"two_parts_{key}"
                    ] = data

        return transplanting_layers_qa_benchmarking

    def _load_modalities_similarities(self) -> Dict[str, Any]:
        """Load modalities similarities data from specific datasets."""
        modalities_similarities = {}

        # Get datasets that contain modalities similarities
        datasets_with_modalities = self.config.get("modalities_similarities", {}).get(
            "datasets_with_modalities", []
        )

        for dataset in datasets_with_modalities:
            # Try loading from main results directory
            dataset_dir = self.results_dir / dataset
            if dataset_dir.exists():
                self._load_modalities_similarities_from_directory(
                    dataset_dir, dataset, modalities_similarities
                )

            # Try loading from coco_captioning results directory if available
            if self.results_coco_captioning_dir is not None:
                coco_dataset_dir = self.results_coco_captioning_dir / dataset
                if coco_dataset_dir.exists():
                    self._load_modalities_similarities_from_directory(
                        coco_dataset_dir, dataset, modalities_similarities
                    )

        return modalities_similarities

    def _load_modalities_similarities_from_directory(
        self, dataset_dir: Path, dataset: str, modalities_similarities: Dict[str, Any]
    ):
        """Helper method to load modalities similarities from a specific directory."""
        # Look for model directories within the dataset
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            model_key = self.config["models"].get(model_name, model_name)

            # Initialize model structure if not exists
            if model_key not in modalities_similarities:
                modalities_similarities[model_key] = {}

            modalities_dir = model_dir / "modalities_similarity"

            if not modalities_dir.exists():
                continue

            # Initialize dataset structure if not exists
            if dataset not in modalities_similarities[model_key]:
                modalities_similarities[model_key][dataset] = {}

            # Load safetensors files
            for file_path in modalities_dir.glob("*.safetensors"):
                filename = file_path.stem

                try:
                    data = load_file(str(file_path))
                except Exception as e:
                    print(f"⚠️  Failed to load {filename}: {e}")
                    continue

                # Extract stream type and similarity type from filename
                # e.g., "output_layer_sample_cosine_similarity" -> "output_layer", "cosine_similarity"
                # e.g., "output_layer_sample_homogeneity_score_euclidean" -> "output_layer", "homogeneity_score_euclidean"
                if "_cosine_similarity" in filename:
                    if "plain" in filename:
                        filename = filename.replace("_plain", "")
                    elif "chat" in filename:
                        filename = filename.replace("_chat", "")

                    stream_type = filename.replace("_sample_cosine_similarity", "")
                    similarity_type = "cosine_similarity"
                elif "_homogeneity_score_" in filename:
                    if "plain" in filename:
                        filename = filename.replace("_plain", "")
                    elif "chat" in filename:
                        filename = filename.replace("_chat", "")

                    # Extract metric from filename (e.g., "euclidean", "cosine")
                    metric = filename.split("_homogeneity_score_")[-1].replace(
                        ".safetensors", ""
                    )
                    if "plain" in metric:
                        metric = metric.replace("_plain", "")
                    elif "chat" in metric:
                        metric = metric.replace("_chat", "")

                    stream_type = filename.replace(
                        f"_sample_homogeneity_score_{metric}", ""
                    )
                    similarity_type = f"homogeneity_score_{metric}"
                elif "_homogeneity_score" in filename:
                    # Legacy support for old format without metric
                    stream_type = filename.replace("_sample_homogeneity_score", "")
                    similarity_type = "homogeneity_score"
                else:
                    # Legacy support for old format
                    stream_type = filename.replace("_sample", "")
                    similarity_type = "cosine_similarity"

                # Initialize stream type structure if not exists
                if stream_type not in modalities_similarities[model_key][dataset]:
                    modalities_similarities[model_key][dataset][stream_type] = {}

                # Store the appropriate tensor based on similarity type
                if (
                    similarity_type == "cosine_similarity"
                    and "layers_cosine_similarity" in data
                ):
                    modalities_similarities[model_key][dataset][stream_type][
                        "cosine_similarity"
                    ] = data["layers_cosine_similarity"]
                elif (
                    similarity_type.startswith("homogeneity_score_")
                    and "layers_homogeneity_scores" in data
                ):
                    modalities_similarities[model_key][dataset][stream_type][
                        similarity_type
                    ] = data["layers_homogeneity_scores"]
                elif (
                    similarity_type == "homogeneity_score"
                    and "layers_homogeneity_scores" in data
                ):
                    # Legacy support for old format
                    modalities_similarities[model_key][dataset][stream_type][
                        "homogeneity_score"
                    ] = data["layers_homogeneity_scores"]
                else:
                    print(f"⚠️  No {similarity_type} data found in {filename}")

        # Filter out raw homogeneity_score entries when metric-specific ones exist
        for model_key in modalities_similarities:
            for dataset in modalities_similarities[model_key]:
                for stream_type in modalities_similarities[model_key][dataset]:
                    stream_data = modalities_similarities[model_key][dataset][
                        stream_type
                    ]

                    # Check if we have metric-specific homogeneity scores
                    has_metric_specific = any(
                        key.startswith("homogeneity_score_")
                        for key in stream_data.keys()
                    )

                    # If we have metric-specific scores, remove the raw homogeneity_score
                    if has_metric_specific and "homogeneity_score" in stream_data:
                        del stream_data["homogeneity_score"]

        return modalities_similarities

    def _load_caption_benchmarking_directory(
        self, directory_name: str
    ) -> Dict[str, Any]:
        """Load caption benchmarking results from a specific directory."""
        caption_benchmarking = {}
        caption_dir = self.results_dir / directory_name

        if not caption_dir.exists():
            print(f"⚠️  No caption benchmarking results found in {directory_name}")
            return caption_benchmarking

        for dataset_dir in caption_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            caption_benchmarking[dataset_name] = {}

            for file_path in dataset_dir.glob("*.csv"):
                filename = file_path.stem

                try:
                    data = pd.read_csv(file_path)
                except Exception as e:
                    print(f"⚠️  Failed to load {filename}: {e}")
                    continue

                # Extract experiment type and parameters
                if "sliding_window" in filename:
                    key = filename.split("sliding_window_")[-1]
                    caption_benchmarking[dataset_name][f"sliding_window_{key}"] = data
                elif "two_parts" in filename:
                    key = filename.split("two_parts_")[-1]
                    caption_benchmarking[dataset_name][f"two_parts_{key}"] = data

        return caption_benchmarking

    def _load_transplanting_layers_caption_benchmarking(self) -> Dict[str, Any]:
        """Load all caption benchmarking experiment results from configured directories."""
        all_caption_benchmarking = {}

        # Get configured caption benchmarking directories
        caption_directories = self.config.get("caption_benchmarking", {}).get(
            "directories", []
        )

        for directory_name in caption_directories:
            directory_data = self._load_caption_benchmarking_directory(directory_name)

            # Merge data with directory name as prefix to avoid conflicts
            for dataset_name, experiments in directory_data.items():
                if dataset_name not in all_caption_benchmarking:
                    all_caption_benchmarking[dataset_name] = {}

                for experiment_name, experiment_data in experiments.items():
                    # Create unique key with directory prefix
                    unique_key = f"{directory_name}_{experiment_name}"
                    all_caption_benchmarking[dataset_name][unique_key] = experiment_data

        return all_caption_benchmarking

    # === Data Access Methods ===

    def get_entropy(
        self,
        dataset: str,
        model: str,
        stream_type: str,
        pooling: str,
    ) -> torch.Tensor:
        """Get dataset entropy data."""
        key = f"{stream_type}_{pooling}"
        try:
            return self.data["model_data_measures"][dataset][model]["dataset_entropy"][
                key
            ]
        except KeyError:
            raise KeyError(f"Entropy data not found: {dataset}/{model}/{key}")

    def get_prompt_entropy(
        self,
        dataset: str,
        model: str,
        stream_type: str,
        pooling: str,
    ) -> torch.Tensor:
        """Get prompt entropy data."""
        key = f"{stream_type}_{pooling}"
        try:
            return self.data["model_data_measures"][dataset][model]["prompt_entropy"][
                key
            ]
        except KeyError:
            raise KeyError(f"Prompt entropy data not found: {dataset}/{model}/{key}")

    def get_intrinsic_dimension(
        self,
        dataset: str,
        model: str,
        stream_type: str,
        pooling: str,
        k: Optional[int] = None,
        range_max: Optional[int] = None,
    ) -> torch.Tensor:
        """Get intrinsic dimension data."""
        if k is None:
            k = self.config["analysis"]["intrinsic_dimension_defaults"]["k"]
        if range_max is None:
            range_max = self.config["analysis"]["intrinsic_dimension_defaults"][
                "range_max"
            ]

        key = f"{stream_type}_{pooling}_k{k}_r{range_max}"
        try:
            return self.data["model_data_measures"][dataset][model][
                "intrinsic_dimension"
            ][key]
        except KeyError:
            raise KeyError(
                f"Intrinsic dimension data not found: {dataset}/{model}/{key}"
            )

    def get_similarity(
        self, dataset: str, stream_type: str, pooling: str, measure: str
    ) -> torch.Tensor:
        """Get similarity data between models."""
        key = f"{stream_type}_{pooling}"
        try:
            return self.data["models_similarities"][dataset][key][measure]
        except KeyError:
            raise KeyError(f"Similarity data not found: {dataset}/{key}/{measure}")

    def get_transplant(self, dataset: str, experiment: str) -> pd.DataFrame:
        """Get transplantation experiment results."""
        try:
            return self.data["transplanting_layers_qa_benchmarking"][dataset][
                experiment
            ]
        except KeyError:
            raise KeyError(f"Transplant data not found: {dataset}/{experiment}")

    def get_modalities_similarity(
        self,
        model: str,
        dataset: str,
        stream_type: str,
        similarity_type: str = "cosine_similarity",
    ) -> torch.Tensor:
        """Get modalities similarity data.

        Args:
            model: Model name
            dataset: Dataset name
            stream_type: Stream type (e.g., "output_layer", "post_mlp")
            similarity_type: Type of similarity ("cosine_similarity", "homogeneity_score",
                           "homogeneity_score_euclidean", or "homogeneity_score_cosine")

        Returns:
            torch.Tensor: Similarity data for the specified type
        """
        try:
            return self.data["modalities_similarities"][model][dataset][stream_type][
                similarity_type
            ]
        except KeyError:
            raise KeyError(
                f"Modalities similarity data not found: {model}/{dataset}/{stream_type}/{similarity_type}"
            )

    def get_caption_benchmarking(
        self, dataset: str, experiment: str, directory_name: str = None
    ) -> pd.DataFrame:
        """Get caption benchmarking experiment results.

        Args:
            dataset: Dataset name (e.g., "coco_captioning")
            experiment: Experiment name (e.g., "sliding_window_ws2_s2")
            directory_name: Optional directory name to specify which experiment to load
                          If None, will try to find the experiment in any available directory
        """
        try:
            if directory_name:
                # Look for specific directory
                full_experiment_name = f"{directory_name}_{experiment}"
                return self.data["transplanting_layers_caption_benchmarking"][dataset][
                    full_experiment_name
                ]
            else:
                # Look for experiment in any available directory
                available_experiments = self.data[
                    "transplanting_layers_caption_benchmarking"
                ][dataset]
                for exp_name, exp_data in available_experiments.items():
                    if exp_name.endswith(f"_{experiment}"):
                        return exp_data

                # If not found, try exact match
                return self.data["transplanting_layers_caption_benchmarking"][dataset][
                    experiment
                ]
        except KeyError:
            raise KeyError(
                f"Caption benchmarking data not found: {dataset}/{experiment}"
            )

    def get_display_name(self, dataset: str) -> str:
        """Get the display name for a dataset."""
        display_names = self.config.get("dataset_display_names", {})
        return display_names.get(dataset, dataset)

    # === Summary Methods ===

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        print(f"📁 Results: {self.results_dir}")
        print(f"🤖 Models: {list(self.config['models'].values())}")
        print(f"📊 Datasets: {self.config['datasets']}")
        print(f"🔧 Stream types: {self.config['analysis']['stream_types']}")
        print(f"🔄 Pooling: {self.config['analysis']['pooling_methods']}")
        print("=" * 50)

    def print_summary(self):
        """Print loaded data summary."""
        print("\n" + "=" * 50)
        print("LOADED DATA SUMMARY")
        print("=" * 50)

        if "model_data_measures" in self.data:
            print("\n📊 MODEL MEASURES:")
            for dataset, models in self.data["model_data_measures"].items():
                print(f"  {dataset}:")
                for model, data in models.items():
                    dataset_entropy_count = len(data.get("dataset_entropy", {}))
                    prompt_entropy_count = len(data.get("prompt_entropy", {}))
                    intrinsic_dimension_count = len(data.get("intrinsic_dimension", {}))
                    print(
                        f"    {model}: {dataset_entropy_count} dataset entropy, "
                        f"{prompt_entropy_count} prompt entropy, "
                        f"{intrinsic_dimension_count} intrinsic dimension"
                    )

        if "models_similarities" in self.data:
            print("\n🤝 SIMILARITIES:")
            for dataset, sims in self.data["models_similarities"].items():
                # Count regular and mean heads projection stream types separately
                regular_streams = [
                    k for k in sims.keys() if not k.startswith("mean_heads_projection")
                ]
                mean_heads_streams = [
                    k for k in sims.keys() if k.startswith("mean_heads_projection")
                ]

                total_configs = len(sims)
                regular_count = len(regular_streams)
                mean_heads_count = len(mean_heads_streams)

                if mean_heads_count > 0:
                    print(
                        f"  {dataset}: {total_configs} configurations ({regular_count} regular + {mean_heads_count} heads mean)"
                    )
                else:
                    print(f"  {dataset}: {total_configs} configurations")

        if "transplanting_layers_qa_benchmarking" in self.data:
            print("\n🔄 TRANSPLANTATION:")
            for dataset, exps in self.data[
                "transplanting_layers_qa_benchmarking"
            ].items():
                print(f"  {dataset}: {list(exps.keys())}")

        if "modalities_similarities" in self.data:
            print("\n🔄 MODALITIES SIMILARITIES:")
            for model, datasets in self.data["modalities_similarities"].items():
                print(f"  {model}:")
                for dataset, stream_types in datasets.items():
                    print(f"    {dataset}:")
                    for stream_type, similarity_types in stream_types.items():
                        print(f"      {stream_type}: {list(similarity_types.keys())}")

        if "transplanting_layers_caption_benchmarking" in self.data:
            print("\n🔄 CAPTION BENCHMARKING:")
            for dataset, exps in self.data[
                "transplanting_layers_caption_benchmarking"
            ].items():
                print(f"  {dataset}: {list(exps.keys())}")

        print("\n" + "=" * 50)


def main(config_path: Union[str, Path] = None) -> ResultsLoader:
    """
    Load all experimental results using configuration.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        ResultsLoader with all data loaded
    """
    loader = ResultsLoader(config_path)
    loader.print_config()
    loader.load_all()
    loader.print_summary()
    return loader


if __name__ == "__main__":
    main()
