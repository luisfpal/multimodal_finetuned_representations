"""CambrianCustom — lmms-eval compatible wrapper using our Cambrian bridge.

The vanilla ``lmms_eval.models.simple.cambrian.Cambrian`` has bugs:
  - ``loglikelihood`` calls ``tokenizer.from_list_format`` (Qwen method, not Cambrian)
  - ``generate_until`` saves images to a hardcoded temp path
  - Loading bypasses our DynamicCache compatibility patch

This module provides ``CambrianCustom`` which:
  - Uses ``utils.cambrian_bridge.load_cambrian_pretrained`` for robust loading
  - Uses our ``cambrian_utils`` for chat template + tokenization + generation
  - Registers as ``"cambrian_custom"`` to avoid collision with the vanilla registry
"""

import logging
from datetime import timedelta
from typing import Any, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import InitProcessGroupKwargs
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from utils.cambrian_bridge import _ensure_cambrian_importable, load_cambrian_pretrained
from utils.cambrian_utils import resolve_cambrian_conv_mode

eval_logger = logging.getLogger("eval_logger")


@register_model("cambrian_custom")
class CambrianCustom(lmms):
    """Cambrian model wrapper for lmms-eval, using our bridge loader."""

    def __init__(
        self,
        pretrained: str = "nyu-visionx/cambrian-8b",
        device: Optional[str] = "cuda",
        device_map: str = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Accelerator for distributed evaluation
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)

        model, tokenizer = load_cambrian_pretrained(
            pretrained, cache_dir=cache_dir, device_map=device_map
        )

        self._model = model
        self._tokenizer = tokenizer
        self.image_processor = getattr(tokenizer, "image_processor", None)
        self._model.eval()

        self.conv_mode = resolve_cambrian_conv_mode(
            model=model, model_name_or_path=pretrained
        )
        self.model_name = pretrained.rstrip("/").split("/")[-1]
        self.batch_size_per_gpu = int(batch_size)

        # Accelerator wrapping for multi-GPU
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # generate_until
    # ------------------------------------------------------------------

    def generate_until(self, requests: List[Instance]) -> List[str]:
        _ensure_cambrian_importable()
        from cambrian.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from cambrian.conversation import conv_templates
        from cambrian.mm_utils import process_images, tokenizer_image_token, tokenizer_image_token_llama3

        use_im_start_end = bool(
            getattr(self.model.config, "mm_use_im_start_end", False)
        )
        tokenize_fn = (
            tokenizer_image_token_llama3
            if self.conv_mode in {"llama_3", "llama_v3"}
            else tokenizer_image_token
        )

        res: list[str] = []

        def _collate(x):
            toks = self._tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                v
                for doc_vis in doc_to_visual
                for v in doc_vis(self.task_dict[task][split][doc_id[0]])
            ]

            gen_kwargs = {**all_gen_kwargs[0]}
            until = [self._tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
            until.append("<|eot_id|>")

            max_new_tokens = gen_kwargs.pop("max_new_tokens", 1024)
            temperature = gen_kwargs.pop("temperature", 0)
            num_beams = gen_kwargs.pop("num_beams", 1)

            # Prepare image
            question = contexts[0]
            if visuals:
                image = visuals[0]
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")

                if use_im_start_end:
                    img_prefix = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
                else:
                    img_prefix = DEFAULT_IMAGE_TOKEN
                qs = f"{img_prefix}\n{question}"

                image_tensor = process_images(
                    [image], self.image_processor, self.model.config
                )
                if isinstance(image_tensor, list):
                    image_tensor = [
                        t.to(self._device, dtype=torch.float16) for t in image_tensor
                    ]
                else:
                    image_tensor = image_tensor.to(
                        self._device, dtype=torch.float16
                    )
                image_sizes = [image.size]
            else:
                qs = question
                image_tensor = None
                image_sizes = None

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenize_fn(
                prompt,
                self._tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(self._device)

            generate_kwargs: dict[str, Any] = {
                "inputs": input_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "num_beams": num_beams,
                "use_cache": True,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
            if image_tensor is not None:
                generate_kwargs["images"] = image_tensor
                generate_kwargs["image_sizes"] = image_sizes

            with torch.inference_mode():
                output_ids = self.model.generate(**generate_kwargs)

            text_output = self._tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()

            for term in until:
                if term:
                    text_output = text_output.split(term)[0]

            res.append(text_output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    # ------------------------------------------------------------------
    # loglikelihood
    # ------------------------------------------------------------------

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        _ensure_cambrian_importable()
        from cambrian.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from cambrian.conversation import conv_templates
        from cambrian.mm_utils import process_images, tokenizer_image_token, tokenizer_image_token_llama3

        use_im_start_end = bool(
            getattr(self.model.config, "mm_use_im_start_end", False)
        )
        tokenize_fn = (
            tokenizer_image_token_llama3
            if self.conv_mode in {"llama_3", "llama_v3"}
            else tokenizer_image_token
        )

        res: list[tuple[float, bool]] = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            continuation = (
                doc_to_target
                if isinstance(doc_to_target, str)
                else doc_to_target(self.task_dict[task][split][doc_id])
            )
            visuals = [
                v
                for v in doc_to_visual(self.task_dict[task][split][doc_id])
            ]

            question = contexts

            # Build prompt with continuation (full) and without (context only)
            if visuals:
                image = visuals[0]
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")

                if use_im_start_end:
                    img_prefix = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
                else:
                    img_prefix = DEFAULT_IMAGE_TOKEN
                qs = f"{img_prefix}\n{question}"

                image_tensor = process_images(
                    [image], self.image_processor, self.model.config
                )
                if isinstance(image_tensor, list):
                    image_tensor = [
                        t.to(self._device, dtype=torch.float16) for t in image_tensor
                    ]
                else:
                    image_tensor = image_tensor.to(
                        self._device, dtype=torch.float16
                    )
                image_sizes = [image.size]
            else:
                qs = question
                image_tensor = None
                image_sizes = None

            # Full conversation with continuation as assistant response
            conv_full = conv_templates[self.conv_mode].copy()
            conv_full.append_message(conv_full.roles[0], qs)
            conv_full.append_message(conv_full.roles[1], continuation)
            full_prompt = conv_full.get_prompt()
            
            # Context-only conversation (no assistant response)
            conv_ctx = conv_templates[self.conv_mode].copy()
            conv_ctx.append_message(conv_ctx.roles[0], qs)
            conv_ctx.append_message(conv_ctx.roles[1], None)
            ctx_prompt = conv_ctx.get_prompt()
            
            full_ids = tokenize_fn(
                full_prompt, self._tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            ctx_ids = tokenize_fn(
                ctx_prompt, self._tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
            )

            if full_ids.dim() == 1:
                full_ids = full_ids.unsqueeze(0)
            if ctx_ids.dim() == 1:
                ctx_ids = ctx_ids.unsqueeze(0)

            full_ids = full_ids.to(self._device)
            ctx_len = ctx_ids.shape[-1]

            # Labels: -100 for context tokens, actual ids for continuation
            labels = full_ids.clone()
            labels[:, :ctx_len] = -100

            forward_kwargs: dict[str, Any] = {
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": torch.ones_like(full_ids),
                "use_cache": False,
            }
            if image_tensor is not None:
                forward_kwargs["images"] = image_tensor
                forward_kwargs["image_sizes"] = image_sizes

            with torch.inference_mode():
                outputs = self.model(**forward_kwargs)

            loss = outputs.loss
            logits = outputs.logits
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = full_ids[:, ctx_len:]
            greedy_tokens = greedy_tokens[:, ctx_len - 1 : full_ids.shape[1] - 1]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    # ------------------------------------------------------------------
    # stubs
    # ------------------------------------------------------------------

    @staticmethod
    def flatten(input_list):
        return [item for sublist in input_list for item in sublist]

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "Multi-round generation not implemented for CambrianCustom"
        )
