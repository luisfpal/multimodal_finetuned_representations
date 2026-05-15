"""LlavaOneVision1_5Custom — lmms-eval compatible wrapper for LLaVA-OneVision-1.5.

The installed lmms-eval v0.4.0 does not include ``llava_onevision1_5``.
This module provides ``LlavaOneVision1_5Custom`` which:
  - Uses ``AutoProcessor`` + ``AutoModelForCausalLM`` (trust_remote_code)
  - Applies the flash-attention compatibility patch before loading
  - Uses ``processor.apply_chat_template`` for prompt formatting
  - Registers as ``"llava_onevision1_5_custom"`` to avoid registry collision
"""

import logging
import math
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from utils.llavaonevision_compat import patch_flash_attn_varlen_import

eval_logger = logging.getLogger("eval_logger")

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


@register_model("llava_onevision1_5_custom")
class LlavaOneVision1_5Custom(lmms):
    """LLaVA-OneVision-1.5 wrapper for lmms-eval."""

    def __init__(
        self,
        pretrained: str = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: str = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        cache_dir: Optional[str] = None,
        max_frames_num: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        # Flash attention compat must be applied before from_pretrained
        patch_flash_attn_varlen_import()

        batch_size = int(batch_size)
        self.batch_size_per_gpu = batch_size
        self.max_frames_num = max_frames_num

        # Accelerator setup (same pattern as InternVL2Custom)
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(
                f"cuda:{accelerator.local_process_index}"
            )
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = "auto"
        else:
            self._device = torch.device(
                f"cuda:{accelerator.local_process_index}"
            )
            self.device_map = f"cuda:{accelerator.local_process_index}"

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.device_map,
            attn_implementation="sdpa",
        )
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained, **load_kwargs
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            pretrained, trust_remote_code=True
        )
        # Left-padding for batch generation
        if hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.padding_side = "left"

        self.model_name = pretrained.rstrip("/").split("/")[-1]

        # Multi-GPU handling
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type. Only DDP, FSDP, DEEPSPEED supported."

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs_ds = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": (
                        self.batch_size_per_gpu * accelerator.num_processes
                    ),
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs_ds
                )

            if accelerator.distributed_type in (
                DistributedType.FSDP,
                DistributedType.DEEPSPEED,
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
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
    def processor(self):
        return self._processor

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
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        return tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # generate_until
    # ------------------------------------------------------------------

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: list[str] = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            # Parse generation kwargs
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
            else:
                until = []

            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            # Remove unsupported kwargs
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("image_aspect_ratio", None)

            max_new_tokens = gen_kwargs.pop("max_new_tokens", 1024)
            gen_kwargs["max_new_tokens"] = max_new_tokens

            # Get visuals
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if not isinstance(visuals, list):
                visuals = [visuals]

            # Build conversation
            content_parts: list[dict] = []
            images: list[Image.Image] = []

            for visual in visuals:
                if isinstance(visual, Image.Image):
                    images.append(visual.convert("RGB"))
                    content_parts.append({"type": "image"})

            content_parts.append({"type": "text", "text": contexts})

            messages = [{"role": "user", "content": content_parts}]

            # Apply chat template
            prompt = self._processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            processor_kwargs = dict(
                text=prompt,
                return_tensors="pt",
                padding=True,
            )
            if images:
                processor_kwargs["images"] = images

            inputs = self._processor(**processor_kwargs)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, **gen_kwargs)

            # Decode only generated tokens
            generated_ids = output_ids[:, input_len:]
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            text_output = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            # Apply until terminators
            for term in until:
                if term:
                    text_output = text_output.split(term)[0]

            res.append(text_output)
            pbar.update(1)

        pbar.close()
        return res

    # ------------------------------------------------------------------
    # loglikelihood
    # ------------------------------------------------------------------

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "loglikelihood not implemented for LlavaOneVision1_5Custom"
        )

    # ------------------------------------------------------------------
    # stubs
    # ------------------------------------------------------------------

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "Multi-round generation not implemented for LlavaOneVision1_5Custom"
        )
