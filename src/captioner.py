import os
import base64
import re
import random
from pathlib import Path
from io import BytesIO
from typing import Optional, Union

import requests
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
from rich.console import Console

console = Console()


class VlmCaptioner:
    """
    A simple and reusable class to generate image captions using a VLM.
    """

    def __init__(self):
        """
        Initializes the VlmCaptioner, loads credentials, and authenticates.
        """
        load_dotenv()
        self.llm_url = os.getenv("ORFEO_LLM_URL")
        self.model_name = os.getenv("ORFEO_MODEL")
        self.jwt_token = self._authenticate()
        if not self.jwt_token:
            raise ValueError("Authentication failed. Please check your .env file.")

        self.client = OpenAI(api_key=self.jwt_token, base_url=self.llm_url)
        console.print("[green]✓ Captioner initialized and authenticated.[/green]")

    def _authenticate(self) -> Optional[str]:
        """
        Authenticates with the Orfeo cluster to get a JWT token.

        Returns:
            The JWT token as a string, or None if authentication fails.
        """
        auth_url = os.getenv("ORFEO_AUTH_URL")
        client_id = os.getenv("ORFEO_CLIENT_ID")
        client_secret = os.getenv("ORFEO_CLIENT_SECRET")
        username = os.getenv("ORFEO_USERNAME")

        if not all([auth_url, client_id, client_secret, username]):
            console.print("[red]Error: Missing authentication variables in .env file.[/red]")
            return None

        try:
            auth_response = requests.post(
                auth_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "password",
                    "client_id": client_id,
                    "username": username,
                    "password": client_secret,
                    "scope": "profile",
                },
            )
            auth_response.raise_for_status()
            return auth_response.json().get("access_token")
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Authentication request failed: {e}[/red]")
            return None

    @staticmethod
    def _encode_image_to_base64(image: Union[Path, Image.Image]) -> str:
        """
        Encodes an image to a base64 data URI.

        Args:
            image: The path to the image file or a PIL Image object.

        Returns:
            The base64-encoded data URI.
        """
        if isinstance(image, (Path, str)):
            img = Image.open(image)
            should_close = True
        else:
            img = image
            should_close = False

        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
        finally:
            if should_close:
                img.close()

    @staticmethod
    def _clean_caption(text: str) -> str:
        """
        Removes reasoning tags (e.g., <think>) from the model's output.

        Args:
            text: The raw output from the model.

        Returns:
            The cleaned caption.
        """
        if not text:
            return ""
        # 1. Remove complete <think>...</think> blocks
        text = re.sub(
            r'<\s*think\s*>.*?</\s*think\s*>',
            '',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

        # 2. Remove any unclosed <think> and everything after it
        text = re.split(
            r'<\s*think\s*>',
            text,
            flags=re.IGNORECASE
        )[0]
        
        return text.strip()

    @staticmethod
    def _remove_incomplete_sentences(text: str) -> str:
        """
        Removes incomplete sentences from the end of the text.
        Keeps text up to the last sentence-ending punctuation (. ! ?).

        Args:
            text: The caption text.

        Returns:
            The text containing only complete sentences.
        """
        if not text:
            return ""
        
        # Regex to find the last occurrence of sentence-ending punctuation
        match = re.search(r'(.*[.!?])', text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def caption_image(
        self,
        image: Union[Path, Image.Image],
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
        max_retries: int = 5,
        seed: int = 42,
    ) -> str:
        """
        Generates a caption for a single image.

        Args:
            image: The path to the image file or a PIL Image object.
            prompt: The text prompt to guide the captioning.
            max_tokens: The maximum number of tokens to generate.
            temperature: The sampling temperature.
            max_retries: Maximum number of retry attempts if validation fails.
            seed: Random seed for reproducibility.

        Returns:
            The generated caption as a string, or an empty string if an error occurs.
        """
        if isinstance(image, (Path, str)):
            image_path = Path(image)
            if not image_path.exists():
                console.print(f"[red]Error: Image file not found at {image_path}[/red]")
                return ""
            image_name = image_path.name
        else:
            image_name = "PIL Image"

        # Set Python's random seed
        random.seed(seed)

        try:
            image_data_uri = self._encode_image_to_base64(image)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert image captioning system. CRITICAL INSTRUCTIONS:\n"
                        "- Detailed thinking off\n"
                        "- Generate exactly 3-5 sentences describing the image\n"
                        "- Start IMMEDIATELY with the visual description\n"
                        "- Output ONLY the final caption - no reasoning, no meta-commentary, no thinking tags\n"
                        "- If you must think, complete your thinking fully before the caption\n"
                        "- Focus on main objects, actions, and scene elements visible in the image\n"
                        "- This is an URGENT, top-priority task - follow these rules exactly"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                },
            ]

            candidates = []

            for attempt in range(max_retries):
                if attempt > 0:
                    console.print(f"[yellow]Retry attempt {attempt + 1}/{max_retries}...[/yellow]")

                # Use seed + attempt to ensure different (but deterministic) results per attempt
                current_seed = seed + attempt
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=current_seed,
                )

                raw_output = response.choices[0].message.content.strip()
                cleaned_output = self._clean_caption(raw_output)
                final_caption = self._remove_incomplete_sentences(cleaned_output)

                # Check for incomplete thinking (has <think> but output suggests it didn't finish)
                raw_lower = raw_output.lower()
                has_open_think = '<think>' in raw_lower
                has_close_think = '</think>' in raw_lower
                has_incomplete_think = has_open_think and not has_close_think
                has_think_in_cleaned = '<think>' in cleaned_output.lower()
                is_empty = not final_caption

                # Accept only if: (1) no thinking tags in cleaned output, (2) not empty, (3) thinking was complete if present
                if not has_think_in_cleaned and not is_empty and not has_incomplete_think:
                    return final_caption

                candidates.append({
                    'caption': final_caption,
                    'has_think': has_think_in_cleaned,
                    'incomplete_think': has_incomplete_think,
                    'raw': raw_output
                })

            console.print("[red]Max retries reached. Selecting best available caption.[/red]")

            # Prioritize: (1) no think tags + complete + non-empty
            clean_candidates = [c for c in candidates
                               if not c['has_think'] and not c['incomplete_think'] and c['caption']]
            if clean_candidates:
                return clean_candidates[-1]['caption']

            # Fallback: (2) complete thinking but has tags
            complete_candidates = [c for c in candidates
                                  if not c['incomplete_think'] and c['caption']]
            if complete_candidates:
                return complete_candidates[-1]['caption']

            # Last resort: (3) any non-empty caption
            any_candidates = [c for c in candidates if c['caption']]
            if any_candidates:
                return any_candidates[-1]['caption']

            return ""

        except Exception as e:
            console.print(f"[red]Error generating caption for {image_name}: {e}[/red]")
            return ""