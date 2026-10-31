import argparse
import json
import os

import numpy as np
import torch
from dadapy import Data

NUM_LAYERS = {
    "onevision_8b_hf": 36,
    "onevision_4b_hf": 36,
    "llava_7b": 32,
    "llava_13b": 40,
    "internvl_8b_hf": 32,
    "cambrian_8b": 32,
}


MODES = {
    "onevision_8b_hf": {
        "coco": ["eoi", "imgavg", "imgeoi", "last", "textavg", "textspec"],
        "imagenet": ["eoi", "imgavg", "last", "textavg", "textspec"],
    },
    "onevision_4b_hf": {
        "coco": ["eoi", "imgavg", "imgeoi", "last", "textavg", "textspec"],
        "imagenet": ["eoi", "imgavg", "last", "textavg", "textspec"],
    },
    "llava_7b": {
        "coco": ["imgavg", "last", "textavg", "textspec"],
    },
    "llava_13b": {
        "coco": ["imgavg", "last", "textavg", "textspec"],
    },
    "internvl_8b_hf": {
        "coco": ["eoi", "imgavg", "imgeoi", "last", "textavg", "textspec"],
        "imagenet": ["eoi", "imgavg", "last", "textavg", "textspec"],
    },
    "cambrian_8b": {
        "coco": ["eoi", "imgavg", "last", "textavg", "textspec"],
        "imagenet": ["eoi", "imgavg", "last", "textavg", "textspec"],
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="results/embedding_similarity",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="onevision_8b_hf",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet_animals_10",
    )
    parser.add_argument(
        "--class_fraction",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = f"{args.data_path}/{args.model_name}/{args.dataset}"
    num_layers = NUM_LAYERS[args.model_name]

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset.startswith("coco"):
        modes = MODES[args.model_name]["coco"]
        siglip_repr_full = torch.load(
            f"{args.data_path}/siglip2_llava_coco.pt"
        )["embeddings"].float().numpy()

        for sim in ["sim85", "sim90"]:
            labels = np.load(f"{args.data_path}/labels_coco_{sim}.npy")
            mask = np.load(f"{args.data_path}/mask_coco_{sim}.npy")
            siglip_repr = siglip_repr_full[mask]
            labels = labels[mask]

            for layer_cpt in ["res2", "norm1"]:
                res_dict_tmp = {}
                res_dict_repr = {}
                for mode in modes:
                    print(f"computing overlap {sim} {layer_cpt} {mode}")
                    results = []
                    results_repr = []
                    for layer in np.arange(0, num_layers, 2):
                        layers = torch.load(
                            f"{base_dir}/layers/l{layer}{layer_cpt}_{mode}.pt"
                        )
                        layers = layers[mask]
                        d = Data(layers.float().numpy())
                        overlap = d.return_label_overlap(
                            labels, class_fraction=args.class_fraction
                        )
                        overlap_repr = [d.return_data_overlap(siglip_repr, k=k) for k in [10, 30, 100]]
                        results.append(overlap)
                        results_repr.append(overlap_repr)
                    res_dict_tmp[mode] = results
                    res_dict_repr[mode] = results_repr

                out_path = f"{args.out_dir}/overlap_{args.dataset}_{layer_cpt}_{sim}_{args.model_name}.json"
                with open(out_path, "w") as f:
                    json.dump(res_dict_tmp, f)
                print(f"saved {out_path}")

                out_path_repr = f"{args.out_dir}/overlap_repr_{args.dataset}_{layer_cpt}_{sim}_{args.model_name}.json"
                with open(out_path_repr, "w") as f:
                    json.dump(res_dict_repr, f)
                print(f"saved {out_path_repr}")
    else:
        modes = MODES[args.model_name]["imagenet"]
        metadata = torch.load(f"{base_dir}/metadata.pt")
        labels = np.array(metadata["class_labels"])
        siglip_repr = torch.load(
            f"{args.data_path}/siglip2_imagenet_animals_10_npc100.pt"
        )["embeddings"].float().numpy()

        for layer_cpt in ["norm1", "res2"]:
            res_dict_tmp = {}
            res_dict_repr = {}
            for mode in modes:
                print(f"computing overlap {layer_cpt} {mode}")
                results = []
                results_repr = []
                for layer in np.arange(0, num_layers, 2):
                    layers = torch.load(
                        f"{base_dir}/layers/l{layer}{layer_cpt}_{mode}.pt"
                    )
                    d = Data(layers.float().numpy())
                    overlap = d.return_label_overlap(
                        labels, class_fraction=args.class_fraction
                    )
                    overlap_repr = [d.return_data_overlap(siglip_repr, k=k) for k in [10, 30, 100]]
                    results.append(overlap)
                    results_repr.append(overlap_repr)

                res_dict_tmp[mode] = results
                res_dict_repr[mode] = results_repr

            out_path = f"{args.out_dir}/overlap_{args.dataset}_{layer_cpt}_{args.model_name}.json"
            with open(out_path, "w") as f:
                json.dump(res_dict_tmp, f)
            print(f"saved {out_path}")

            out_path_repr = f"{args.out_dir}/overlap_repr_{args.dataset}_{layer_cpt}_{args.model_name}.json"
            with open(out_path_repr, "w") as f:
                json.dump(res_dict_repr, f)
            print(f"saved {out_path_repr}")


if __name__ == "__main__":
    main()
