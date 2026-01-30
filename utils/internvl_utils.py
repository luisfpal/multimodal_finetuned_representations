from typing import Tuple, List, Union, Optional
from PIL import Image
import copy


import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import PreTrainedModel

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# InternVL2 special tokens
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_CONTEXT_TOKEN_ID = 92546


def build_transform(input_size: int) -> T.Compose:
    """Build image transformation pipeline."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Find best tile grid matching aspect ratio."""
    best_diff, best_ratio = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]
        ):
            best_diff, best_ratio = diff, ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """Tile image dynamically based on aspect ratio."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j == n
        },
        key=lambda x: x[0] * x[1],
    )
    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size
    )
    target_w, target_h = image_size * best_ratio[0], image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_w, target_h))
    tiles = []
    for i in range(blocks):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size,
        )
        tiles.append(resized_img.crop(box))

    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def get_pixel_values(
    images: Union[Image.Image, List[Image.Image]],
    image_size: int = 448,
    max_tiles: int = 12,
) -> Tuple[torch.Tensor, List[int]]:
    # Normalize to list
    if isinstance(images, Image.Image):
        images = [images]

    all_tiles = []
    num_patches_list = []

    for image in images:
        tiles = dynamic_preprocess(
            image, image_size=image_size, max_num=max_tiles, use_thumbnail=True
        )
        num_patches_list.append(len(tiles))
        all_tiles.extend(tiles)

    # Stack all tiles into single tensor
    transform = build_transform(image_size)
    pixel_values = torch.stack([transform(t) for t in all_tiles])
    return pixel_values.to(torch.bfloat16), num_patches_list


def apply_chat_template(
    model: PreTrainedModel,
    text: Union[str, List[str]],
    num_patches_list: Optional[List[int]] = None,
    guide_text: str = "",
) -> List[str]:
    """Apply InternVL chat template to text, optionally with image tokens.

    Args:
        model: InternVL model (needed for conv_template)
        text: Single string or list of strings
        num_patches_list: Number of patches per image. If None, text-only mode (no image tokens).
    """
    # Normalize to list
    if isinstance(text, str):
        text = [text]

    model.eval()
    queries = []

    for idx, txt in enumerate(text):
        if guide_text and guide_text != "":
            txt = txt + "\n " + guide_text + "\n"
        
        # Add <image> placeholder only if we have images
        has_image = num_patches_list is not None and idx < len(num_patches_list)
        if has_image and "<image>" not in txt:
            txt = "<image>\n" + txt

        # Build conversation template
        template = copy.deepcopy(model.conv_template)
        template.system_message = model.system_message
        template.messages = []
        template.append_message(template.roles[0], txt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        # Replace <image> with IMG_CONTEXT tokens (only if we have images)
        if has_image:
            num_patches = num_patches_list[idx]
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        queries.append(query)
        
    return queries


def get_combined_embeddings(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Replace IMG_CONTEXT token embeddings with vision features."""
    model.eval()
    device = model.device
    input_ids = input_ids.to(device)

    # Extract vision features for ALL tiles
    with torch.no_grad():
        vit_embeds = model.extract_feature(pixel_values.to(device))

    # Get text embeddings
    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape

    # Flatten for replacement
    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)

    # Find and replace IMG_CONTEXT positions
    selected = input_ids_flat == IMG_CONTEXT_TOKEN_ID
    input_embeds_flat[selected] = vit_embeds.reshape(-1, C).to(input_embeds_flat.device)

    return input_embeds_flat.reshape(B, N, C)


def generate_responses(
    model: PreTrainedModel,
    tokenizer,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 50,
) -> List[str]:
    """Generate text responses from combined embeddings."""
    template = copy.deepcopy(model.conv_template)
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    with torch.no_grad():
        output = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
        )

    responses = tokenizer.batch_decode(output, skip_special_tokens=True)
    return [r.split(template.sep.strip())[0].strip() for r in responses]

