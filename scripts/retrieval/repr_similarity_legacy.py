"""LEGACY/EXPLORATORY: pseudolabel-clustering analysis, __main__ block was already'
commented out upstream — kept for provenance/reference only, not wired to any launcher."""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

embeddings_images = torch.load("results/embedding_similarity/siglip2_llava_coco.pt")

embedding_text = torch.load(
    "results/embedding_similarity/qwen3_8b_llava_descriptions.pt"
)
assert embedding_text["embeddings"].shape[0] == embeddings_images["embeddings"].shape[0]


def cluster_by_similarity(embeds, sim_threshold=0.85, min_cluster_size=20):
    embeds = F.normalize(embeds, dim=1).numpy()
    dist = embeds @ embeds.T
    adj = (dist > sim_threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    _, labels = connected_components(csr_matrix(adj), directed=False, connection="weak")
    counts = Counter(labels)
    mask = np.array([counts[l] > min_cluster_size for l in labels])
    return labels, mask


img_labels, img_mask = cluster_by_similarity(
    embeddings_images["embeddings"], sim_threshold=0.85
)
txt_labels, txt_mask = cluster_by_similarity(
    embedding_text["embeddings"], sim_threshold=0.95
)


def cluster_balance(labels_kept):
    """Normalized Shannon entropy of cluster sizes in [0, 1].
    1.0 = perfectly even, 0 = single cluster dominates."""
    sizes = np.array(list(Counter(labels_kept).values()), dtype=np.float64)
    p = sizes / sizes.sum()
    H = -np.sum(p * np.log(p))
    return H / np.log(len(sizes)) if len(sizes) > 1 else 0.0


print(f"image dataset size {np.sum(img_mask)}")
print(f"num img clusters {len(np.unique(img_labels[img_mask]))}")
print(f"img cluster balance {cluster_balance(img_labels[img_mask]):.3f}")
img_counts = sorted(list(Counter(img_labels[img_mask]).values()))[::-1]
plt.hist(img_counts)

print(f"txt dataset size {np.sum(txt_mask)}")
print(f"num txt clusters {len(np.unique(txt_labels[txt_mask]))}")
print(f"txt cluster balance {cluster_balance(txt_labels[txt_mask]):.3f}")
txt_counts = Counter(txt_labels[txt_mask]).values()
plt.hist(txt_counts)


# Intersection: points kept in a >20 cluster in BOTH modalities.
# Two points share a joint cluster iff they share both image and text label.
both_mask = img_mask & txt_mask
joint_labels = np.full(len(both_mask), -1, dtype=np.int64)
pair_to_id = {}
for i in np.where(both_mask)[0]:
    key = (int(img_labels[i]), int(txt_labels[i]))
    if key not in pair_to_id:
        pair_to_id[key] = len(pair_to_id)
    joint_labels[i] = pair_to_id[key]


# Prune joint clusters with <= 20 samples.
joint_size = Counter(joint_labels[both_mask])
small_joint = {l for l, c in joint_size.items() if c <= 20 or c >= 400}
drop = both_mask & np.isin(joint_labels, list(small_joint))
both_mask = both_mask & ~drop
joint_labels[drop] = -1

print(f"joint dataset size {np.sum(both_mask)}")
print(f"num joint clusters {len(np.unique(joint_labels[both_mask]))}")
print(f"joint cluster balance {cluster_balance(joint_labels[both_mask]):.3f}")
joint_counts = Counter(joint_labels[both_mask]).values()
plt.hist(joint_counts)

list(joint_counts)


np.save("results/embedding_similarity/mask_coco_qwen3_8b_siglip2.npy", both_mask)
np.save("results/embedding_similarity/pseudolabels_coco_qwen3_8b_siglip2.npy", joint_labels)


# # d.return_label_overlap(layers["targets"].numpy(), class_fraction=0.2)
# results = []
# d = Data(F.normalize(layers["embeddings"], dim=1).numpy())
# metadata = torch.load(
#     "results/embedding_similarity/onevision_4b_hf/imagenet_animals_10/metadata.pt"
# )

# for layer in np.arange(0, 36):
#     print(layer)
#     layers_llava = torch.load(
#         f"results/embedding_similarity/onevision_4b_hf/imagenet_animals_10/layers/l{layer}res2_last_image_only_no_assistant.pt"
#     )
#     d = Data(layers_llava.float().numpy())
#     overlap_x2 = d.return_label_overlap(np.array(metadata["class_labels"]), k=30)
#     results.append(overlap_x2)


# np.array(results)

# plt.plot(np.array(results))
# print(metadata)
# metadata.keys()
# metadata["class_labels"]
# np.array(metadata["class_labels"])[
#     np.where(np.array(metadata["predicted_token_ids"]) == 32)[0]
# ]

# len(np.where(np.array(metadata["predicted_token_ids"]) == 32)[0])
# print(metadata["predicted_tokens"])
# np.unique(metadata["imagenet_codes"])


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         default="/home/diego/Documents/area_science/giskard/conversation-embeddings/datasets",
#     )
#     parser.add_argument(
#         "--data_name",
#         type=str,
#         default="cnn_daily",
#     )
#     parser.add_argument("--model_name_or_path", type=str, default="./tokenizer")
#     parser.add_argument(
#         "--use_flash_attn",
#         action="store_true",
#         help="If passed, will use flash attention to train the model.",
#     )
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument(
#         "--out_dir", type=str, default=None, help="Where to store the final model."
#     )
#     parser.add_argument("--filename", type=str, default="")
#     parser.add_argument(
#         "--seed", type=int, default=None, help="A seed for reproducible training."
#     )
#     parser.add_argument(
#         "--preprocessing_num_workers",
#         type=int,
#         default=6,
#         help="The number of processes to use for the preprocessing.",
#     )
#     parser.add_argument(
#         "--avg_token",
#         action="store_true",
#         help="If passed, ID will be measured on the last token represenation.",
#     )
#     parser.add_argument(
#         "--save_distances",
#         action="store_true",
#         help="If passed, the distance matrices will be saved",
#     )
#     parser.add_argument(
#         "--remove_duplicates",
#         action="store_true",
#         help="If passed, duplicate datapoints will be removed",
#     )
#     parser.add_argument(
#         "--target_layer",
#         type=str,
#         default="norm1",
#         help="The name of the layer to analyze.",
#     )
#     parser.add_argument(
#         "--similar_captions_n_prompts",
#         type=int,
#         default=50,
#         help="First N dataset captions used as queries for per-layer neighbor JSON.",
#     )
#     parser.add_argument(
#         "--similar_captions_k",
#         type=int,
#         default=10,
#         help="Top-K most similar captions (cosine) per query at each layer.",
#     )
#     parser.add_argument(
#         "--similar_captions_json_suffix",
#         type=str,
#         default="_similar_captions.json",
#         help="Suffix for per-layer JSON files (after l{layer_id}{filename_stem}).",
#     )
#     parser.add_argument(
#         "--max_num_samples",
#         type=int,
#         default=None,
#         help="Suffix for per-layer JSON files (after l{layer_id}{filename_stem}).",
#     )
#     parser.add_argument(
#         "--k",
#         type=int,
#         nargs="+",
#         default=[10],
#         help="k values for k-NN overlap vs Qwen embeddings (e.g. --k 5 10 20).",
#     )
#     parser.add_argument(
#         "--force_recompute",
#         action="store_true",
#         help="If passed, recompute representations even if saved files already exist.",
#     )
#     parser.add_argument(
#         "--embedder",
#         type=str,
#         default="qwen0.6b",
#         choices=["qwen0.6b", "qwen8b"],
#         help="Which Qwen embedder was used to produce the reference embeddings.",
#     )
#     parser.add_argument(
#         "--gaussian_cka_sigma",
#         type=float,
#         nargs="+",
#         default=[0.1],
#         help="RBF bandwidth scale passed to gram_rbf(..., threshold=sigma) for Gaussian-kernel CKA.",
#     )
#     parser.add_argument(
#         "--metrics_json",
#         type=str,
#         default="",
#         help="Basename for the combined metrics JSON under out_dir/model/coco_image_question. "
#         "Default: {target_layer}_qwen06b_llava13b_all_metrics.json",
#     )
#     parser.add_argument(
#         "--detail_json",
#         type=str,
#         default="",
#         help="Path to detail_23k.json. When set, uses this dataset instead of COCO-QA.",
#     )
#     parser.add_argument(
#         "--image_dir",
#         type=str,
#         default="",
#         help="Directory containing the images referenced in --detail_json.",
#     )
#     args = parser.parse_args([])
#     return args


# def first_prompts_topk_neighbors_json(
#     embeddings: torch.Tensor,
#     texts: List[str],
#     n_prompts: int = 50,
#     k: int = 10,
# ) -> list:
#     """Cosine similarity in representation space; excludes self from neighbors."""
#     n_unique = embeddings.shape[0]
#     n_q = min(n_prompts, n_unique)
#     k_eff = min(k, max(0, n_unique - 1))
#     if k_eff == 0:
#         return [{"index": i, "prompt": texts[i], "neighbors": []} for i in range(n_q)]

#     emb = F.normalize(embeddings.float(), dim=1)
#     sim = emb[:n_q] @ emb.T

#     out = []
#     for i in range(n_q):
#         row = sim[i].clone()
#         row[i] = -float("inf")
#         vals, idx = torch.topk(row, k=k_eff)
#         neighbors = [
#             {
#                 "index": int(j),
#                 "prompt": texts[int(j)],
#                 "cosine_similarity": float(v),
#             }
#             for j, v in zip(idx.tolist(), vals.tolist())
#         ]
#         out.append({"index": i, "prompt": texts[i], "neighbors": neighbors})
#     return out


# # def main():


# args = parse_args()

# args.out_dir = "results/embedding_similarity"
# args.embedder = "qwen8b"

# embedder_filenames = {
#     "qwen0.6b": f"qwen3_0.6b_llava_descriptions.pt",
#     "qwen8b": f"qwen3_8b_llava_descriptions.pt",
# }
# qwen_path = os.path.join(args.out_dir, embedder_filenames[args.embedder])
# qwen_embeddings_dict = torch.load(qwen_path, map_location="cpu")
# qwen_embeddings = qwen_embeddings_dict["embeddings"]

# torch.norm(qwen_embeddings, dim=1)


# # qwen_embeddings = qwen_embeddings.to("cuda").double()
# # cosine_similarity = torch.nn.functional.cosine_similarity(
# #     qwen_embeddings, qwen_embeddings.unsqueeze(1), dim=1, eps=1e-8
# # )


# def compute_cluster_similarity(reference, layer_path, k=40):
#     print("computing reference")
#     reference = F.normalize(reference.float(), dim=1)
#     data = Data(qwen_embeddings.numpy().astype(np.float64))
#     data.compute_distances(maxk=100)
#     id_list, id_error_list, id_distance_list = data.return_id_scaling_gride()
#     data.set_id(id_list[2])
#     log_den, log_den_error = data.compute_density_kNN(k=k)
#     reference_labels = data.compute_clustering_ADP(Z=1.65, halo=False)

#     # reference = F.normalize(reference.float(), dim=1)
#     for idx in [0, 10, 20, 30]:
#         print(f"computing layer {idx}")
#         layer = torch.load(f"{layer_path}/l{idx}res2.pt", map_location="cpu")
#         data = Data(layer.double().numpy().astype(np.float64))
#         data.compute_distances(maxk=100)
#         id_list, id_error_list, id_distance_list = data.return_id_scaling_gride()
#         data.set_id(id_list[2])
#         log_den, log_den_error = data.compute_density_kNN(k=k)
#         layer_labels = data.compute_clustering_ADP(Z=1.65, halo=False)
#         print(data.N_clusters)

#         ari = metrics.adjusted_rand_score(reference_labels, layer_labels)
#         hom = metrics.homogeneity_score(reference_labels, layer_labels)
#         com = metrics.completeness_score(reference_labels, layer_labels)
#         print(f"layer {idx}")
#         print("ari", ari)
#         print("homogeneity", hom)
#         print("completeness", com)


# layer_path = (
#     "results/embedding_similarity/onevision_4b_hf/coco_detailed_descriptions/layers"
# )
# compute_cluster_similarity(qwen_embeddings, layer_path, k=40)


# print(f"target layer: {args.target_layer}")

# simplify_name = {
#     "llava-hf/llava-1.5-7b-hf": "llava_7b_hf",
#     "llava-hf/llava-1.5-13b-hf": "llava_13b_hf",
#     "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct": "onevision_4b_hf",
#     "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": "onevision_8b_hf",
# }
# num_layers_dict = {
#     "llava-hf/llava-1.5-7b-hf": 32,
#     "llava-hf/llava-1.5-13b-hf": 40,
#     "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct": 36,
#     "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": 36,
# }
# model_name = simplify_name[args.model_name_or_path]
# n_layer = num_layers_dict[args.model_name_or_path]
# dirpath = os.path.join(args.out_dir, model_name, "coco_detailed_descriptions", "layers")

# # The saver writes each block as `l{layer_id}{target_layer}.pt` for
# # layer_id in 0..n_layer-1; for norm1/norm2 it also writes a final
# # `l{n_layer}{target_layer}.pt` for the post-stack norm.
# target_layer_labels = list(range(n_layer))
# if args.target_layer in ("norm1", "norm2"):
#     target_layer_labels.append(n_layer)

# layer_paths = {
#     lid: os.path.join(dirpath, f"l{lid}{args.target_layer}.pt")
#     for lid in target_layer_labels
# }
# act_by_id = {
#     lid: torch.load(layer_paths[lid], map_location="cpu") for lid in target_layer_labels
# }

# print("computing metrics")
# overlaps_by_layer: dict[int, list[float]] = {}
# overlaps_by_layer_norm: dict[int, list[float]] = {}
# imbalances: dict[int, tuple[float, float]] = {}
# imbalances_norm: dict[int, tuple[float, float]] = {}
# cka_by_layer: dict[int, dict[float, float]] = {}
# cka_norm_by_layer: dict[int, dict[float, float]] = {}
# linear_cka_by_layer: dict[int, float] = {}
# linear_cka_norm_by_layer: dict[int, float] = {}

# qwen_norm = F.normalize(qwen_embeddings.float(), dim=1)
# gram_qwen_by_sigma = {
#     sigma: gram_rbf(qwen_embeddings, sigma) for sigma in args.gaussian_cka_sigma
# }
# gram_qwen_norm_by_sigma = {
#     sigma: gram_rbf(qwen_norm, sigma) for sigma in args.gaussian_cka_sigma
# }


# for layer_id in target_layer_labels:
#     print(layer_id)
#     act = act_by_id[layer_id]

#     overlaps_by_layer[layer_id] = compute_overlap_torch(
#         qwen_embeddings, act, ks=tuple(args.k)
#     )
#     i12, i21 = compute_imbalance_torch(qwen_embeddings, act, exact_rank=False)
#     imbalances[layer_id] = (i12, i21)

#     cka_by_layer[layer_id] = {
#         sigma: float(cka(gram_qwen_by_sigma[sigma], gram_rbf(act, sigma)))
#         for sigma in args.gaussian_cka_sigma
#     }
#     linear_cka_by_layer[layer_id] = float(
#         feature_space_linear_cka(qwen_embeddings, act)
#     )

#     act_n = F.normalize(act, dim=1)
#     overlaps_by_layer_norm[layer_id] = compute_overlap_torch(
#         qwen_embeddings, act_n, ks=tuple(args.k)
#     )
#     i12n, i21n = compute_imbalance_torch(qwen_embeddings, act_n, exact_rank=False)
#     imbalances_norm[layer_id] = (i12n, i21n)

#     cka_norm_by_layer[layer_id] = {
#         sigma: float(cka(gram_qwen_norm_by_sigma[sigma], gram_rbf(act_n, sigma)))
#         for sigma in args.gaussian_cka_sigma
#     }
#     linear_cka_norm_by_layer[layer_id] = float(
#         feature_space_linear_cka(qwen_norm, act_n)
#     )

# filename = f"{args.target_layer}_{args.embedder}_{model_name}"
# metrics_basename = args.metrics_json.strip() or f"{filename}_all_metrics.json"
# metrics_path = os.path.join(dirpath, metrics_basename)

# hyperparameters = {
#     "overlap_k": list(args.k),
#     "gaussian_cka_sigma": list(args.gaussian_cka_sigma),
#     "max_num_samples": args.max_num_samples,
# }

# metrics: dict = {}
# for ki, k_val in enumerate(args.k):
#     metrics[f"overlap__k_{k_val}"] = {
#         str(lid): overlaps_by_layer[lid][ki] for lid in target_layer_labels
#     }
#     metrics[f"overlap_norm__k_{k_val}"] = {
#         str(lid): overlaps_by_layer_norm[lid][ki] for lid in target_layer_labels
#     }

# metrics["imbalance"] = {
#     str(lid): {"qwen_to_llava": imbalances[lid][0], "llava_to_qwen": imbalances[lid][1]}
#     for lid in target_layer_labels
# }
# metrics["imbalance_norm"] = {
#     str(lid): {
#         "qwen_to_llava": imbalances_norm[lid][0],
#         "llava_to_qwen": imbalances_norm[lid][1],
#     }
#     for lid in target_layer_labels
# }

# for sigma in args.gaussian_cka_sigma:
#     sig_key = str(sigma)
#     metrics[f"gaussian_cka__sigma_{sig_key}"] = {
#         str(lid): cka_by_layer[lid][sigma] for lid in target_layer_labels
#     }
#     metrics[f"gaussian_cka_norm__sigma_{sig_key}"] = {
#         str(lid): cka_norm_by_layer[lid][sigma] for lid in target_layer_labels
#     }

# metrics["linear_cka"] = {
#     str(lid): linear_cka_by_layer[lid] for lid in target_layer_labels
# }
# metrics["linear_cka_norm"] = {
#     str(lid): linear_cka_norm_by_layer[lid] for lid in target_layer_labels
# }

# payload = {"hyperparameters": hyperparameters, "metrics": metrics}
# with open(metrics_path, "w", encoding="utf-8") as f:
#     json.dump(payload, f, ensure_ascii=False, indent=2)
# print(f"Saved combined metrics to {metrics_path}")

# # if __name__ == "__main__":

# #     main()
