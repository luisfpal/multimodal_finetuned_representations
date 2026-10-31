import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec


def format_size(imb, thr_small=0.2, thr_medium=0.3, base_size=50, radius_factor=3):
    if imb < thr_small:
        return base_size
    elif imb < thr_medium:
        return base_size / radius_factor
    else:
        return base_size / radius_factor**2


def format(
    imbalance_dict,
    label,
    color,
    marker,
    invert=False,
    base_size=100,
    radius_factor=4,
    thr_small=0.2,
    thr_medium=0.3,
    thr_large=0.5,
    zorder=1,
    width=0.7,
):

    x = []
    y = []
    size = []
    linewidths = []

    # list of possible marker sizes (3: thr_large, thr_medium, thr_small imbalance)
    sizes = np.sort(
        list(
            set(
                [
                    format_size(
                        imb,
                        thr_small=thr_small,
                        thr_medium=thr_medium,
                        base_size=base_size,
                        radius_factor=radius_factor,
                    )
                    for imb in np.linspace(10**-5, 1, 10000)
                ]
            )
        )
    )

    # map the size of the marker to obtain an increasing line width
    # size_to_width = {sizes[0]: width, sizes[1]: width, sizes[2]: width}

    for layer1 in imbalance_dict.keys():
        for layer2 in imbalance_dict[layer1].keys():
            imbalance = imbalance_dict[layer1][layer2]
            if imbalance > thr_large:
                # no marker is imbalance is too thr_large
                continue
            else:
                if invert:
                    x.append(layer2)
                    y.append(layer1)

                else:
                    x.append(layer1)
                    y.append(layer2)

                size_tmp = format_size(
                    imbalance,
                    thr_small=thr_small,
                    thr_medium=thr_medium,
                    base_size=base_size,
                    radius_factor=radius_factor,
                )
                size.append(size_tmp)
                # all have the same width
                linewidths.append(width)
                # linewidths.append(size_to_width[size_tmp])

    # labels to be put in the legend
    size_to_label = {
        sizes[2]: f"\\subseteq [0, {thr_small:.2f}]",
        sizes[1]: f"\\subseteq [{thr_small:.2f}, {thr_medium:.2f}]",
        sizes[0]: f"\\subseteq [{thr_medium:.2f}, {thr_large:.2f}]",
    }

    # split the datasets according to forward/backwards and marker sizes
    datasets = []
    for size_tmp in sizes:
        indices = np.nonzero(np.array(size) == size_tmp)[0]
        if len(indices) > 0:

            datasets.append(
                {
                    "x": np.array(x)[indices],
                    "y": np.array(y)[indices],
                    "marker": marker,
                    "size": np.array(size)[indices],
                    "linewidths": np.array(linewidths)[indices],
                    "edgecolor": color,
                    "label": f"$\Delta({label}) {size_to_label[size_tmp]}$",
                    "zorder": zorder,
                }
            )

    return datasets


def get_layers(model1, model2, datasets, base_path):
    # here we just extract the number of layers of the models
    # and double check they are consistent across the datasets

    layers1 = None
    layers2 = None
    for text_data in datasets:
        try:
            filename = f"inf_imb_{text_data.lower()}_{model1.lower()}_{model2.lower()}_hidden_from_first.pickle"
            with open(f"{base_path}/{filename}", "rb") as f:
                first_dict = pickle.load(f)
        except:
            filename = f"{model1.lower()}_{model2.lower()}_from_first.pickle"
            with open(f"{base_path}/{filename}", "rb") as f:
                first_dict = pickle.load(f)

        if layers1 is None and layers2 is None:
            layers1 = np.array(list(first_dict.keys()))
            layers2 = np.array(list(first_dict[1].keys()))
        else:
            assert np.all(layers1 == np.array(list(first_dict.keys())))
            assert np.all(layers2 == np.array(list(first_dict[1].keys())))
    return layers1, layers2


def get_imbalance_10k_new(model1, model2, datasets, direction, base_path):

    assert direction in ["from_first", "from_second"]
    layers1, layers2 = get_layers(
        model1=model1, model2=model2, datasets=datasets, base_path=base_path
    )

    # initialize imbalance matrix
    imbalance = np.zeros((len(layers1), len(layers2)))

    filename = f"{model1.lower()}_{model2.lower()}_{direction}.pickle"
    with open(f"{base_path}/{filename}", "rb") as f:
        first_dict = pickle.load(f)

    for i, layer1 in enumerate(first_dict.keys()):
        for j, layer2 in enumerate(first_dict[layer1].keys()):
            imbalance[i, j] = np.mean(first_dict[layer1][layer2])

    imbalance_dict = defaultdict(dict)
    for i, lay1 in enumerate(layers1):
        for j, lay2 in enumerate(layers2):
            imbalance_dict[lay1][lay2] = imbalance[i, j]
    return imbalance_dict


def get_tick_peaks(is_peak):

    ticks_to_select = np.nonzero(((is_peak - 1) % 3) == 0)[0]
    tick_list = list(is_peak[ticks_to_select])

    if len(tick_list) == 0 and len(is_peak) == 1:
        if is_peak[0] % 3 == 1:
            tick_list.append(is_peak[0] - 1)
        else:
            tick_list.append(is_peak[0] + 1)
    else:
        if min((is_peak - 1)) % 3 == 1:
            tick_list.append(min(tick_list) - 1)
        if max((is_peak - 1)) % 3 == 2:
            tick_list.append(max(tick_list) + 1)

    return tick_list


# ***********************************************************************************

base_path = "./results/imbalance"
fix_thr = True


def get_formatted_imbalance(imbalance):
    layers1 = imbalance.shape[0]
    layers2 = layers1
    imbalance_dict = defaultdict(dict)
    for i, lay1 in enumerate(range(layers1)):
        for j, lay2 in enumerate(range(layers2)):
            imbalance_dict[lay1][lay2] = imbalance[i, j]
    return imbalance_dict


def prepare_datasets(fix_thr, imbalance_first, imbalance_second):

    if fix_thr:
        thr_small = 0.022
        thr_medium = 0.049
        thr_large = 0.058
        # thr_small = 0.02
        # thr_medium = 0.05
        # thr_large = 0.08
    else:
        all_imbalances = []
        for lay, val in imbalance_first.items():
            all_imbalances.extend(list(val.values()))
        for lay, val in imbalance_second.items():
            all_imbalances.extend(list(val.values()))

        percentiles = np.percentile(np.array(all_imbalances), q=[4, 20, 30])
        thr_small = percentiles[0]
        thr_medium = percentiles[1]
        thr_large = percentiles[2]

    datasets = format(
        imbalance_first,
        label="text -> img",
        color="C1",
        marker="o",
        base_size=50,
        radius_factor=5,
        thr_small=thr_small,
        thr_medium=thr_medium,
        thr_large=thr_large,
        # zorder=100,
        width=0.8,
    )

    datasets.extend(
        format(
            imbalance_second,
            invert=False,
            label="img -> text",
            color="C0",
            marker="s",
            base_size=50,
            radius_factor=5,
            thr_small=thr_small,
            thr_medium=thr_medium,
            thr_large=thr_large,
            # zorder=1,
        )
    )
    return datasets


tasks = ["vqav2", "cocoqa", "gqa", "mmbench", "mme"]
datasets = {}
for mode in ["pre", "ft"]:
    img_text = []
    text_img = []
    for data in tasks:
        img_text.append(
            np.expand_dims(np.load(f"{base_path}/{data}/{mode}_ii_img_txt.npy"), axis=0)
        )
        text_img.append(
            np.expand_dims(np.load(f"{base_path}/{data}/{mode}_ii_txt_img.npy"), axis=0)
        )

    img_text = np.concatenate(img_text, axis=0).mean(axis=0)
    text_img = np.concatenate(text_img, axis=0).mean(axis=0)

    imbalance_dict_first = get_formatted_imbalance(text_img)
    imbalance_dict_second = get_formatted_imbalance(img_text)

    datasets[mode] = prepare_datasets(
        fix_thr=True,
        imbalance_first=imbalance_dict_first,
        imbalance_second=imbalance_dict_second,
    )


# Create a Seaborn plot

sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

fig = plt.figure(figsize=(8, 4.2))
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[0])

for data in datasets["pre"]:
    sns.scatterplot(
        ax=ax,
        x=data["x"],
        y=data["y"],
        marker=data["marker"],
        s=data["size"],
        facecolors="None",
        edgecolors=data["edgecolor"],
        linewidths=data["linewidths"],
        label=data["label"],
        # zorder=100,
    )  # zorder ensures scatterplot is above circle patch

# Create scatterplot with seaborn
ticks = np.arange(0, 32, 3) + 1
ax.legend(fontsize=7)
ax.set_xticks(ticks)
ax.set_xlim(0.4, 33)
ax.set_yticks(ticks)
ax.set_ylim(0.4, 33)
ax.set_xlabel("text input", fontsize=11)
ax.set_ylabel("visual input", fontsize=11)
ax.set_title("pretrained", size=12)


ax = fig.add_subplot(gs[1])
for data in datasets["ft"]:
    sns.scatterplot(
        ax=ax,
        x=data["x"],
        y=data["y"],
        marker=data["marker"],
        s=data["size"],
        facecolors="None",
        edgecolors=data["edgecolor"],
        linewidths=data["linewidths"],
        label=data["label"],
        # zorder=100,
    )  # zorder ensures scatterplot is above circle patch

# Create scatterplot with seaborn
ticks = np.arange(0, 32, 3) + 1
ax.legend(fontsize=7)
ax.set_xticks(ticks)
ax.set_xlim(0.4, 33)
ax.set_yticks(ticks)
ax.set_ylim(0.4, 33)
ax.set_xlabel("text input", fontsize=11)
ax.set_ylabel("visual input", fontsize=11)
ax.set_title("finetuned", size=12)

fig.tight_layout()


data = np.load(f"{base_path}/vqav2/pre_ii_txt_img.npy")
plt.plot(data[10, :])

# if fix_thr:
#     folder = f"fix_thr_{thr_small}_{thr_medium}_{thr_large}"
# else:
#     folder = "fix_percentiles"

# results_dir = f"./figures/final/10k/{folder}"
# os.makedirs(results_dir, exist_ok=True)
# plt.savefig(f"{results_dir}/avg_data_{model1.lower()}_{model2.lower()}_striped_10k.pdf")
# plt.savefig(
#     f"{results_dir}/avg_data_{model1.lower()}_{model2.lower()}_striped_10k.png",
#     dpi=200,
# )
