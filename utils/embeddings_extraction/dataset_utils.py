"""Dataset / data-loading helpers used by embeddings_vlms.py.

Centralizes the per-family Dataset classes, record builders, the legacy
text-only chat template, and the batch collators.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image

from utils.embeddings_extraction.caption_starters import starters
from utils.embeddings_extraction.imagenet_loader import ImageFolder
from utils.internvl_conversations import get_conv_template
from utils.internvl_utils import get_pixel_values as internvl_get_pixel_values


def return_caption(captions):
    return ". ".join(s.strip().rstrip(".") for s in captions) + "."


LLAVA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] + ' '}}{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)

# Plain-Python equivalent of LLAVA_CHAT_TEMPLATE.
# Each message is {"role": str, "content": list[{"type": "text"|"image", "text": str}]}.
# System messages are rendered without the role prefix.
# For every message: images (<image>\n) come first, then text tokens (each followed by a space).
# The {% generation %}...{% endgeneration %} markers in the Jinja template only affect
# tokenization boundaries for training; in plain text they produce the same output as
# non-assistant text, so the branch below is identical for both roles.
#
# def apply_llava_chat_template(
#     messages: list[dict],
#     add_generation_prompt: bool = True,
# ) -> str:
#     result = ""
#     for message in messages:
#         if message["role"] != "system":
#             result += message["role"].upper() + ": "
#         for content in message["content"]:          # images first
#             if content["type"] == "image":
#                 result += "<image>\n"
#         for content in message["content"]:          # text next
#             if content["type"] == "text":
#                 result += content["text"] + " "
#     if add_generation_prompt:
#         result += "ASSISTANT:"
#     return result


def process_data(inputs, tokenizer):
    descriptions = [return_caption(captions) for captions in inputs["captions"]]
    questions = [
        qas[0]["question"].strip().rstrip("?") + "?"
        for qas in inputs["questions_and_answers"]
    ]

    conversations = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": description + "\n" + question}],
            }
        ]
        for description, question in zip(descriptions, questions)
    ]

    formatted_inputs = [
        tokenizer.apply_chat_template(
            conversation,
            chat_template=LLAVA_CHAT_TEMPLATE,
            add_generation_prompt=True,
            tokenize=False,
        )
        for conversation in conversations
    ]
    # Equivalent plain-string version of the template above (single user turn, text only):
    # formatted_inputs = [
    #     f"USER: {description}\n{question} ASSISTANT:"
    #     for description, question in zip(descriptions, questions)
    # ]

    tokenized = tokenizer(formatted_inputs, add_special_tokens=False)

    return {
        "text_prompts": formatted_inputs,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def load_and_validate_detail23k(
    json_path: str,
    image_dir: str,
    with_descriptions: bool = False,
    validate_images: bool = True,
) -> List[Dict]:
    """Load detail_23k.json and return one record per entry.

    Each record contains ``{image_path, question}`` and, when
    ``with_descriptions=True``, the assistant's detailed description under
    ``description``. ``validate_images=False`` skips the image-existence
    check (used by the text-only path, which never opens the images).
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if validate_images:
        missing = [
            os.path.join(image_dir, f"COCO_train2014_{entry["image"]}")
            for entry in data
            if not os.path.isfile(os.path.join(image_dir, f"COCO_train2014_{entry["image"]}"))
        ]
        if missing:
            sample = "\n  ".join(missing[:10])
            tail = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"{len(missing)} image(s) missing from {image_dir}:\n  {sample}{tail}"
            )

    records = []
    for entry in data:
        human_turn = next(c for c in entry["conversations"] if c["from"] == "human")
        # strip the <image> placeholder that LLaVA annotations embed in the text
        question = human_turn["value"].replace("<image>", "").strip()
        record = {
            "image_path": os.path.join(image_dir, f"COCO_train2014_{entry["image"]}"),
            "question": question,
        }
        if with_descriptions:
            gpt_turn = next(c for c in entry["conversations"] if c["from"] == "gpt")
            record["description"] = gpt_turn["value"].strip()
        records.append(record)
    return records


def build_coco_karpathy_records(
    image_dir: str,
    split: str = "test",
    validate_images: bool = True,
) -> List[Dict]:
    """Load yerevann/coco-karpathy and return one record per entry.

    Each record contains ``image_path`` (``image_dir + filename``),
    ``filename``, the original ``captions`` list, and ``description`` (the
    captions concatenated via ``return_caption``: trailing periods stripped,
    joined with ``". "``, final period appended). ``question`` is left empty;
    image-mode runs use the captions only via ``--text_only`` or skip text
    entirely via ``--image_only``.
    """
    from datasets import load_dataset

    ds = load_dataset("yerevann/coco-karpathy", split=split)

    records = []
    missing = []
    for entry in ds:
        filename = entry["filename"]
        image_path = os.path.join(image_dir, filename)
        if validate_images and not os.path.isfile(image_path):
            missing.append(image_path)
            continue
        captions = list(entry["sentences"])
        records.append({
            "image_path": image_path,
            "filename": filename,
            "captions": captions,
            "description": return_caption(captions),
            "question": "",
        })

    if missing:
        sample = "\n  ".join(missing[:10])
        tail = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} image(s) missing from {image_dir}:\n  {sample}{tail}"
        )
    return records


def default_winoground_cache_dir() -> str:
    """Default location for the per-PNG winoground cache.

    Prefers ``$HF_HOME/winoground_images`` so the cache lives next to the HF
    datasets cache (same filesystem, same backup policy); falls back to
    ``~/.cache/huggingface/winoground_images``.
    """
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(hf_home, "winoground_images")


def build_winoground_records(
    cache_dir: str,
    split: str = "test",
    cache_images: bool = True,
) -> List[Dict]:
    """Load facebook/winoground and flatten to one record per (entry, side).

    Each winoground entry has two images (image_0, image_1) and two captions
    (caption_0, caption_1). This emits 2 records per entry in side-major
    order: ``[(id=0, side=0), (id=0, side=1), (id=1, side=0), ...]``, so the
    saved [2N, D] tensor reshapes cleanly to [N, 2, D].

    With ``cache_images=True`` (default), PIL images are dumped to ``cache_dir``
    as PNGs (skipped if already present) so the existing image_path-based
    dataset classes work unchanged. With ``cache_images=False`` (text-only
    runs), no PIL decode and no disk writes happen — image_path is still set
    to the would-be cache location for traceability.

    Per-record fields: ``image_path, filename, caption, description (=caption),
    question (""), entry_id, side, tag, secondary_tag``.
    """
    from datasets import load_dataset

    if cache_images:
        os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset("facebook/winoground", split=split)

    records = []
    for entry in ds:
        eid = entry["id"]
        for side in (0, 1):
            caption = entry[f"caption_{side}"]
            filename = f"winoground_{eid:04d}_{side}.png"
            image_path = os.path.join(cache_dir, filename)
            if cache_images and not os.path.isfile(image_path):
                # PIL access is the trigger for HF's image-feature decode;
                # gating it on cache_images keeps text-only runs from
                # decoding 800 PNGs we never use.
                img = entry[f"image_{side}"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(image_path)
            records.append({
                "image_path": image_path,
                "filename": filename,
                "caption": caption,
                "description": caption,
                "question": "",
                "entry_id": eid,
                "side": side,
                "tag": entry.get("tag", ""),
                "secondary_tag": entry.get("secondary_tag", ""),
            })
    return records


CAPTIONING_PROMPT = "Describe the image."


def build_imagenet_records(
    image_dir: str,
    class_indices=None,
    num_classes=None,
    num_images_per_class=None,
    seed=0,
):
    """Build per-image records (image_path + captioning question + class metadata)
    from an ImageNet-style directory. Each image in a class receives a starter
    from caption_starters.starters via round-robin on the in-class index, so
    every class uses each starter an equal number of times (within the remainder
    when num_images_per_class is not a multiple of len(starters))."""
    kwargs = dict(root=image_dir, nimg_cat=num_images_per_class)
    if class_indices is not None:
        kwargs["class_indices"] = class_indices
    else:
        kwargs["nclasses"] = num_classes
        kwargs["seed"] = seed
    folder = ImageFolder(**kwargs)

    idx_to_class = {v: k for k, v in folder.class_to_idx.items()}
    records = []
    per_class_counter: Dict[int, int] = {}
    for path, target in folder.samples:
        i = per_class_counter.get(target, 0)
        per_class_counter[target] = i + 1
        starter_idx = i % len(starters)
        starter = starters[starter_idx]
        question = (
            f'{CAPTIONING_PROMPT} Please start the description with "{starter}".'
        )
        records.append({
            "image_path": path,
            "question": question,
            "class_label": target,
            "imagenet_code": idx_to_class[target],
            "starter": starter,
            "starter_idx": starter_idx,
        })
    return records, folder


class DetailImageDataset(torch.utils.data.Dataset):
    """Load & process detail_23k samples on the fly inside DataLoader workers.

    Replaces datasets.map which (a) ran sequentially with num_proc=1 because
    the processor is not fork-safe via HF Arrow writers, and (b) breaks Arrow's
    column-length invariant when LLaVA-OneVision's dynamic tiling makes
    pixel_values vary in shape across samples. Parallelism comes from the
    DataLoader's own worker pool (num_workers).
    """

    def __init__(
        self,
        records: List[Dict],
        processor,
        image_only: bool = False,
        add_generation_prompt: bool = True,
    ):
        self.records = records
        self.processor = processor
        self.image_only = image_only
        self.add_generation_prompt = add_generation_prompt
        conversations = []
        for r in records:
            content = [{"type": "image"}]
            if not image_only:
                content.append({"type": "text", "text": r["question"]})
            conversations.append([{"role": "user", "content": content}])
        # Chat-template strings are image-independent (processor expands the
        # <image> placeholder later), so we precompute them once up front.
        self.text_prompts = [
            processor.apply_chat_template(
                c, add_generation_prompt=add_generation_prompt, tokenize=False
            )
            for c in conversations
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image = Image.open(self.records[idx]["image_path"]).convert("RGB")
        encoded = self.processor(
            text=self.text_prompts[idx],
            images=image,
            return_tensors="pt",
        )
        # Forward every tensor the processor produced so model-specific
        # metadata (image_grid_thw, image_sizes, image_flags, ...) survives
        # to the model. input_ids / attention_mask shed their leading batch
        # dim so the collator can repad them; image-side tensors stay intact
        # and get concatenated along dim 0 later.
        item = {"text_prompts": self.text_prompts[idx]}
        for key, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key in ("input_ids", "attention_mask"):
                item[key] = value[0]
            else:
                item[key] = value
        return item


class InternVLImageDataset(torch.utils.data.Dataset):
    """Dataset for InternVL2.

    InternVL2 has no unified HF Processor: the chat template in
    tokenizer_config.json expects ``message['content']`` to be a string and
    crashes on the list-of-dicts content that DetailImageDataset builds.
    The reference impl (model.chat in modeling_internvl_chat.py) builds the
    prompt manually with the ``internlm2-chat`` conversation template, then
    expands a single ``<image>`` placeholder into
    ``<img><IMG_CONTEXT>*num_image_token*N</img>`` where N = number of 448x448
    tiles produced by dynamic_preprocess for that specific image.

    We replicate that here: build the prompt per-sample (since the IMG_CONTEXT
    repeat count depends on the per-image tile count), tile the image with
    load_image, and tokenize the resulting plain string. Returns input_ids /
    attention_mask / pixel_values / image_flags shaped to match what the
    InternVL2 forward expects (image_flags is required, not optional — it's
    .squeeze(-1)'d unconditionally inside forward).
    """

    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    PLACEHOLDER = "<image>"

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        num_image_token: int,
        image_only: bool = False,
        add_generation_prompt: bool = True,
        conv_template_name: str = "internlm2-chat",
        max_num_tiles: int = 6,
        input_size: int = 448,
        pixel_values_dtype: torch.dtype = torch.bfloat16,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token
        self.image_only = image_only
        self.add_generation_prompt = add_generation_prompt
        self.conv_template_name = conv_template_name
        self.max_num_tiles = max_num_tiles
        self.input_size = input_size
        # Must match the model's loaded weight dtype: ToTensor returns float32
        # but the InternViT patch_embedding conv2d weights are bfloat16, so a
        # mismatch raises "Input type (float) and bias type (c10::BFloat16)".
        self.pixel_values_dtype = pixel_values_dtype
        # Human-readable per-sample prompts saved as metadata. Keep the
        # <image> placeholder in place rather than the expanded IMG_CONTEXT
        # span so the saved string is readable and tile-count-independent.
        self.text_prompts = [
            self._build_query_with_placeholder(r.get("question", ""))
            for r in records
        ]

    def _build_query_with_placeholder(self, question: str) -> str:
        template = get_conv_template(self.conv_template_name)
        user_msg = (
            self.PLACEHOLDER
            if self.image_only
            else f"{self.PLACEHOLDER}\n{question}"
        )
        template.append_message(template.roles[0], user_msg)
        if self.add_generation_prompt:
            template.append_message(template.roles[1], None)
        return template.get_prompt()

    def __len__(self):
        return len(self.records)

    def _build_query(self, question: str, n_tiles: int) -> str:
        template = get_conv_template(self.conv_template_name)
        user_msg = self.PLACEHOLDER if self.image_only else f"{self.PLACEHOLDER}\n{question}"
        template.append_message(template.roles[0], user_msg)
        if self.add_generation_prompt:
            template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = (
            self.IMG_START_TOKEN
            + self.IMG_CONTEXT_TOKEN * self.num_image_token * n_tiles
            + self.IMG_END_TOKEN
        )
        return query.replace(self.PLACEHOLDER, image_tokens, 1)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["image_path"]).convert("RGB")
        pixel_values, num_patches_list = internvl_get_pixel_values(
            image, image_size=self.input_size, max_tiles=self.max_num_tiles
        )
        pixel_values = pixel_values.to(self.pixel_values_dtype)  # (n_tiles, 3, input_size, input_size)
        n_tiles = num_patches_list[0]

        query = self._build_query(record.get("question", ""), n_tiles)
        enc = self.tokenizer(query, return_tensors="pt")

        # image_flags: 1 per kept tile. forward() does image_flags.squeeze(-1)
        # then vit_embeds[image_flags == 1], so shape must be (n_tiles, 1).
        image_flags = torch.ones((n_tiles, 1), dtype=torch.long)

        return {
            "text_prompts": query,
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }


class TextOnlyDataset(torch.utils.data.Dataset):
    """Text-only dataset: feeds the detail_23k assistant 'detailed
    description' as the user message, rendered through each model's chat
    template with no image. Used by --text_only.

    Per sample, emits ``input_ids`` / ``attention_mask`` for the full
    chat-templated string (with the trailing assistant role header) plus
    a ``userlast_pos`` scalar — the index of the last description token in
    the full input, immediately before the assistant header begins. The
    extractor pools at that position into the ``user_last`` extra so the
    description's terminal-token rep can be saved separately from the
    standard "last input token" rep (which lands on the assistant
    scaffold itself).
    """

    FAMILY_LLAVA_HF = "llava_hf"
    FAMILY_ONEVISION = "onevision"
    FAMILY_INTERNVL = "internvl"
    FAMILY_CAMBRIAN = "cambrian"

    def __init__(
        self,
        records: List[Dict],
        processor,
        family: str,
        cambrian_conv_mode: str = None,
        add_generation_prompt: bool = True,
    ):
        self.records = records
        self.processor = processor
        self.family = family
        self.cambrian_conv_mode = cambrian_conv_mode
        self.add_generation_prompt = add_generation_prompt
        # The HF processors (LLaVA-hf, OneVision) expose .tokenizer; for
        # InternVL2 / Cambrian the "processor" we hold *is* the tokenizer.
        self.tokenizer = (
            processor.tokenizer
            if family in (self.FAMILY_LLAVA_HF, self.FAMILY_ONEVISION)
            else processor
        )
        if family == self.FAMILY_CAMBRIAN:
            assert cambrian_conv_mode is not None, (
                "cambrian_conv_mode is required for Cambrian text-only"
            )
        self.text_prompts = [
            self._build_prompt(r["description"], with_assistant=add_generation_prompt)
            for r in records
        ]

    def _build_prompt(self, description: str, with_assistant: bool) -> str:
        if self.family in (self.FAMILY_LLAVA_HF, self.FAMILY_ONEVISION):
            conv = [
                {"role": "user", "content": [{"type": "text", "text": description}]}
            ]
            return self.processor.apply_chat_template(
                conv, add_generation_prompt=with_assistant, tokenize=False,
            )
        if self.family == self.FAMILY_INTERNVL:
            tmpl = get_conv_template("internlm2-chat")
            tmpl.append_message(tmpl.roles[0], description)
            if with_assistant:
                tmpl.append_message(tmpl.roles[1], None)
            return tmpl.get_prompt()
        if self.family == self.FAMILY_CAMBRIAN:
            from cambrian.conversation import conv_templates
            conv = conv_templates[self.cambrian_conv_mode].copy()
            conv.append_message(conv.roles[0], description)
            if with_assistant:
                conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        raise ValueError(f"Unknown text-only family: {self.family}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        description = self.records[idx]["description"]
        full = self._build_prompt(description, with_assistant=self.add_generation_prompt)
        # Same conversation but stop before the assistant role header. The
        # last token of this prefix tokenization marks userlast_pos in the
        # full input. (For chat templates that just append the assistant
        # scaffold to the end, the user-only render is a strict string
        # prefix of the full render, and BPE/SentencePiece tokenizers are
        # stable on prefixes up to a 1-token boundary jitter — clamped
        # below for safety.)
        user_only = self._build_prompt(description, with_assistant=False)
        full_enc = self.tokenizer(full, return_tensors="pt")
        user_only_enc = self.tokenizer(user_only, return_tensors="pt")
        L_full = full_enc.input_ids.shape[1]
        L_user = user_only_enc.input_ids.shape[1]
        userlast = max(0, min(L_user - 1, L_full - 1))
        return {
            "text_prompts": full,
            "input_ids": full_enc.input_ids[0],
            "attention_mask": full_enc.attention_mask[0],
            # 1-d tensor so DataCollatorForVisionLM's torch.cat along dim 0
            # produces a [B] tensor across the batch.
            "userlast_pos": torch.tensor([userlast], dtype=torch.long),
        }


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict:

        input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
        attention_mask = [
            torch.tensor(instance["attention_mask"]) for instance in instances
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0, padding_side="left"
        )

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


@dataclass
class DataCollatorForVisionLM:
    """Collator for multimodal (image + text) batches from process_detail_data."""
    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(inst["input_ids"]) for inst in instances],
            batch_first=True, padding_value=self.pad_token_id, padding_side="left",
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(inst["attention_mask"]) for inst in instances],
            batch_first=True, padding_value=0, padding_side="left",
        )
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Every other tensor field (pixel_values, image_grid_thw, image_sizes,
        # image_flags, ...) is concatenated along dim 0. This is the right op
        # for: LLaVA-1.5 pixel_values [1,C,H,W] -> [B,C,H,W]; Qwen2-VL flat
        # pixel_values [P_i,D] -> [sum P_i,D]; InternVL2 tiled pixel_values
        # [n_tiles_i,C,H,W] -> [sum n_tiles_i,C,H,W]; and all the per-image
        # metadata tensors ([N_i,3], [N_i,2], ...) shaped the same way.
        skip = {"input_ids", "attention_mask", "text_prompts"}
        for key, value in instances[0].items():
            if key in skip or not isinstance(value, torch.Tensor):
                continue
            batch[key] = torch.cat(
                [torch.as_tensor(inst[key]) for inst in instances], dim=0
            )
        return batch
