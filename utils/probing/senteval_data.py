"""Provides utilities for loading SentEval probing task data and a collator for causal language modeling batches."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import Dataset

TASK_FILE_NAMES = {
    "SentLen": "sentence_length.txt",
    "WC": "word_content.txt",
    "TreeDepth": "tree_depth.txt",
    "TopConst": "top_constituents.txt",
    "BShift": "bigram_shift.txt",
    "Tense": "past_present.txt",
    "SubjNum": "subj_number.txt",
    "ObjNum": "obj_number.txt",
    "SOMO": "odd_man_out.txt",
    "CoordInv": "coordination_inversion.txt",
}


def load_senteval_data(
    task_name: str,
    data_dir: str,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load data from the SentEval repository.

    Args:
        task_name: SentEval probing task name.
        data_dir: Directory containing the SentEval probing data.

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
        Labels are encoded as integers.
    """

    assert (
        task_name in TASK_FILE_NAMES
    ), f"{task_name} not found. Available task names: {TASK_FILE_NAMES.keys()}"
    filename = TASK_FILE_NAMES[task_name]

    with open(f"{data_dir}/{filename}", "r") as f:
        lines = f.readlines()

    dataset = {
        "tr": {"text": [], "label": []},
        "va": {"text": [], "label": []},
        "te": {"text": [], "label": []},
    }

    for line in lines:
        split, label, text = line.strip().split("\t")
        dataset[split]["text"].append(text.strip())
        dataset[split]["label"].append(label)

    assert set(dataset["va"]["label"]) == set(dataset["te"]["label"])
    assert set(dataset["tr"]["label"]) == set(dataset["te"]["label"])
    # Fit label encoder on all labels
    all_labels = dataset["tr"]["label"]

    label_to_int = {label: i for i, label in enumerate(set(all_labels))}

    train_labels = [label_to_int[label] for label in dataset["tr"]["label"]]
    val_labels = [label_to_int[label] for label in dataset["va"]["label"]]
    test_labels = [label_to_int[label] for label in dataset["te"]["label"]]

    dataset["va"]["label"] = val_labels
    dataset["te"]["label"] = test_labels
    dataset["tr"]["label"] = train_labels

    train_dataset = Dataset.from_dict(dataset["tr"])
    test_dataset = Dataset.from_dict(dataset["te"])
    val_dataset = Dataset.from_dict(dataset["va"])

    return train_dataset, test_dataset, val_dataset


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int
    padding_side: str = "right"

    def __call__(self, instances: Sequence[Dict]) -> Dict:

        input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
        attention_mask = [
            torch.tensor(instance["attention_mask"]) for instance in instances
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side=self.padding_side,
        )

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
