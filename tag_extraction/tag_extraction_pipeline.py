"""
Comprehensive Tag Extraction Pipeline.

A generic pipeline for extracting ALL visual semantics from images as tags,
then classifying them as relevant or irrelevant based on the task.

Pipeline Stages:
    Stage 1: Image → VLM → Comprehensive Tags (per image)
    Stage 2: Aggregate all tags → Unique vocabulary (dataset-level)
    Stage 3: LLM classifies tags as relevant/irrelevant (once, task-based)
    Stage 4: Apply mapping to generate final CSV (per image)

Key Features:
    - Configurable VLM (LLaVA, Qwen-VL, InternVL)
    - Configurable LLM (GPT-4, Llama, local models)
    - Task-based relevance (no class names needed)
    - Optional human review step
    - No tag merging (preserves all granularity)

Usage:
    from tag_extraction import TagExtractionPipeline

    pipeline = TagExtractionPipeline(config)
    pipeline.run(
        image_paths=image_paths,
        image_indices=indices,
        image_targets=targets,
        task_description="car type classification"
    )
"""

import os
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import csv

import torch
from tqdm import tqdm
from PIL import Image


# ============================================
# Configuration
# ============================================


@dataclass
class TagExtractionConfig:
    """Configuration for tag extraction pipeline."""

    # ---- VLM Settings ----
    vlm_model: str = "llava"  # "llava", "qwen-vl", "internvl"
    vlm_model_path: str = "llava-hf/llava-1.5-7b-hf"
    vlm_device: str = "cuda"

    # ---- LLM Settings ----
    llm_model: str = "llama3"  # "llama3", "gpt-4", "gpt-3.5-turbo", "ollama"
    llm_model_path: str = "meta-llama/Meta-Llama-3-70B-Instruct"
    llm_api_key: str = ""  # For OpenAI
    llm_base_url: str = ""  # For local/custom endpoints

    # ---- Processing ----
    vlm_batch_size: int = 1  # VLMs typically process one at a time
    llm_tag_batch_size: int = 100  # Tags per LLM call
    min_tag_frequency: int = 5  # Ignore rare tags

    # ---- Task ----
    task_description: str = ""

    # ---- Human Review ----
    enable_human_review: bool = False

    # ---- Output ----
    output_dir: str = "./tag_extraction_output"

    # ---- Resume ----
    resume_from_stage: int = 1  # 1, 2, 3, or 4


# ============================================
# VLM Prompts
# ============================================

# VLM_PROMPTS = {
#     "objects": (
#         "List ALL objects and their parts visible in this image. "
#         "For each object, also include its specific type, category, breed, species, model, or variant if identifiable. "
#         "Be as specific as possible (e.g., not just 'bird' but 'seagull', not just 'flower' but 'rose', not just 'dog' but 'golden retriever'). "
#         "Include main subjects, background objects, and small details. "
#         "Output ONLY as comma-separated tags, nothing else. "
#         "Example format: golden retriever, dog, tail, paw, oak tree, tree, trunk, leaves, fire hydrant"
#     ),
#     "object_details": (
#         "For each main object in this image, describe its fine-grained characteristics: "
#         "specific type/category/breed/species/model/variant, distinguishing features, "
#         "sub-parts, and any identifiable markings or characteristics. "
#         "Be as specific and detailed as possible about WHAT each object is. "
#         "Output ONLY as comma-separated tags, nothing else. "
#         "Example format: labrador retriever, adult dog, floppy ears, short coat, black nose, long tail, four legs"
#     ),
#     "attributes": (
#         "List ALL visual attributes in this image: colors (be specific like 'navy blue' not just 'blue'), "
#         "sizes (relative and absolute), shapes, materials, textures, surface properties, "
#         "conditions, states, patterns, and any other visual qualities. "
#         "Output ONLY as comma-separated tags, nothing else. "
#         "Example format: navy blue, bright red, large, small, metallic, glossy, rusty, worn, striped, checkered, smooth, rough"
#     ),
#     "scene": (
#         "Describe the scene, location, setting, and environment as tags. "
#         "Include specific type of place, indoor/outdoor, urban/rural/natural, "
#         "geographic hints, time period indicators, and environmental context. "
#         "Output ONLY as comma-separated tags, nothing else. "
#         "Example format: beach, coastal, sandy shore, ocean view, tropical, outdoor, daytime, summer"
#     ),
#     # "actions": (
#     #     "List all actions, poses, movements, states, behaviors, or interactions. "
#     #     "Include what objects/subjects are doing and how they relate to each other. "
#     #     "Output ONLY as comma-separated tags, nothing else. "
#     #     "Example format: running, jumping, sitting, eating, sleeping, flying, swimming, parked, moving, open, closed"
#     # ),
#     # "photo_style": (
#     #     "Describe the photo/image characteristics: lighting conditions, weather, "
#     #     "camera angle, perspective, photo style, image quality, focus, depth of field. "
#     #     "Output ONLY as comma-separated tags, nothing else. "
#     #     "Example format: natural lighting, sunny, overcast, close-up, aerial view, wide angle, shallow depth of field, high resolution, professional"
#     # ),
#     "additional": (
#         "List ANY other visual details not yet mentioned: text, logos, symbols, brands, "
#         "numbers, signs, subtle details, partially visible objects, reflections, shadows. "
#         "Look carefully for small or easy-to-miss details. "
#         "Output ONLY as comma-separated tags, nothing else. "
#         "If nothing additional, output: none"
#     ),
# }

VLM_PROMPTS = {
    "objects": (
        "List ALL objects and their parts visible in this image. "
        "For each object, also include its specific type, category, breed, species, model, or variant if identifiable. "
        "Be as specific as possible (e.g., not just 'bird' but 'seagull', not just 'flower' but 'rose', not just 'dog' but 'golden retriever'). "
        "Include main subjects, background objects, and small details. "
        # "Output ONLY as comma-separated tags, nothing else. "
        # "Example format: golden retriever, dog, tail, paw, oak tree, tree, trunk, leaves, fire hydrant"
        "For each main object in this image, describe its fine-grained characteristics: "
        "specific type/category/breed/species/model/variant, distinguishing features, "
        "sub-parts, and any identifiable markings or characteristics. "
        "Be as specific and detailed as possible about WHAT each object is. "
        # "Output ONLY as comma-separated tags, nothing else. "
        # "Example format: labrador retriever, adult dog, floppy ears, short coat, black nose, long tail, four legs"
        "Describe the scene, location, setting, and environment as tags. "
        "Include specific type of place, indoor/outdoor, urban/rural/natural, "
        "geographic hints, time period indicators, and environmental context. "
        # "Output ONLY as comma-separated tags, nothing else. "
        # "Example format: beach, coastal, sandy shore, ocean view, tropical, outdoor, daytime, summer"
        "List ANY other visual details not yet mentioned: text, logos, symbols, brands, "
        "numbers, signs, subtle details, partially visible objects, reflections, shadows. "
        "Look carefully for small or easy-to-miss details. "
        "Output ONLY as comma-separated tags, nothing else. "
    ),
}


# ============================================
# LLM Prompt for Relevance Classification
# ============================================

LLM_RELEVANCE_PROMPT = """You are an expert at analyzing visual concepts for machine learning bias detection.

TASK: {task_description}

YOUR TASK:
Given the task above, classify each visual tag as RELEVANT or IRRELEVANT.

RELEVANT tags are visual concepts that:
1. Are intrinsic properties of the subject being classified
2. Would naturally vary WITH the class and help distinguish between classes
3. Are CAUSED BY or are PART OF the subject being classified
4. Would still be visible if you isolated ONLY the subject from its environment

IRRELEVANT tags are visual concepts that:
1. Are properties of the environment, background, or context
2. Are NOT intrinsic to the subject being classified
3. Co-occur with subjects but are NOT caused by the subject
4. Could introduce spurious correlations (bias)
5. Include: backgrounds, weather, lighting, camera properties, co-occurring objects, locations, time of day, photographer style

REASONING APPROACH:
For each tag, ask yourself:
"If I showed ONLY the subject being classified with no background or context, would this tag still apply?"
- If YES → RELEVANT (it's intrinsic to the subject)
- If NO → IRRELEVANT (it's context/environment)

EXAMPLES FOR "car type classification":
- "sedan" → RELEVANT (describes a car type)
- "wheel" → RELEVANT (intrinsic car part)
- "red" → RELEVANT (color of the car itself)
- "metallic" → RELEVANT (car's material/finish)
- "4 doors" → RELEVANT (car's property)
- "parking lot" → IRRELEVANT (location, not car property)
- "sunny" → IRRELEVANT (weather)
- "tree" → IRRELEVANT (background)
- "road" → IRRELEVANT (environment)
- "person" → IRRELEVANT (co-occurring object)

EXAMPLES FOR "bird species classification":
- "bird" → RELEVANT (the subject)
- "feathers" → RELEVANT (intrinsic bird part)
- "yellow beak" → RELEVANT (bird's attribute)
- "small" → RELEVANT (bird's size)
- "flying" → RELEVANT (bird's action/state)
- "water" → IRRELEVANT (environment)
- "branch" → IRRELEVANT (perch, not part of bird)
- "forest" → IRRELEVANT (habitat/background)
- "sunny" → IRRELEVANT (lighting condition)

EXAMPLES FOR "facial expression recognition":
- "smile" → RELEVANT (expression)
- "frown" → RELEVANT (expression)
- "eyes" → RELEVANT (facial feature)
- "teeth showing" → RELEVANT (part of expression)
- "indoor" → IRRELEVANT (location)
- "office" → IRRELEVANT (background)
- "shirt" → IRRELEVANT (clothing, not face)
- "bright lighting" → IRRELEVANT (photo condition)

TAGS TO CLASSIFY:
{tag_list}

OUTPUT FORMAT (respond with valid JSON only, no other text):
{{
    "relevant": ["tag1", "tag2", ...],
    "irrelevant": ["tag3", "tag4", ...],
    "reasoning": {{
        "tag1": "brief reason",
        "tag3": "brief reason"
    }}
}}

IMPORTANT:
- Classify ALL tags - every tag must appear in exactly one list
- When uncertain, lean towards IRRELEVANT (safer for bias prevention)
- The reasoning field should include at least a few examples for verification
- Output ONLY valid JSON, no markdown, no explanation before or after"""


# ============================================
# Base VLM Class
# ============================================


class BaseVLM(ABC):
    """Base class for Vision-Language Models."""

    @abstractmethod
    def query(self, image: Image.Image, prompt: str) -> str:
        """Query the VLM with an image and prompt."""
        pass

    def extract_tags(self, image: Image.Image) -> Dict[str, List[str]]:
        """Extract comprehensive tags from an image."""
        all_tags = []
        category_tags = {}

        for category, prompt in VLM_PROMPTS.items():
            response = self.query(image, prompt)
            tags = self._parse_tags(response)
            category_tags[category] = tags
            all_tags.extend(tags)

        # Deduplicate while preserving order
        seen = set()
        unique_tags = []
        for tag in all_tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return {"all_tags": unique_tags, **category_tags}

    def _parse_tags(self, response: str) -> List[str]:
        """Parse comma-separated tags from VLM response."""
        # Clean up response
        response = response.strip()

        # Handle "none" or empty responses
        if not response or response.lower() in ["none", "n/a", "nothing", "none."]:
            return []

        # Split by comma
        tags = [t.strip().lower() for t in response.split(",")]

        # Filter empty and clean up
        tags = [t for t in tags if t and len(t) > 0 and t not in ["none", "n/a"]]

        # Remove any trailing periods or extra punctuation
        tags = [re.sub(r"[.!?]+$", "", t).strip() for t in tags]

        # Filter again after cleaning
        tags = [t for t in tags if t and len(t) > 0]

        return tags


# ============================================
# LLaVA VLM
# ============================================


class LLaVAVLM(BaseVLM):
    """LLaVA Vision-Language Model."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            print(f"Loading LLaVA from {model_path}...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model.eval()
            print("LLaVA loaded successfully.")

        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def query(self, image: Image.Image, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            images=[image], text=[text_prompt], padding=True, return_tensors="pt"
        ).to(self.device, torch.float16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )

        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract assistant response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        elif "assistant" in response.lower():
            parts = response.lower().split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
                # Remove leading colon or newline
                response = re.sub(r"^[:\s]+", "", response)

        return response


# ============================================
# Qwen-VL VLM (supports both Qwen-VL and Qwen2-VL)
# ============================================


class QwenVLVLM(BaseVLM):
    """Qwen-VL and Qwen2-VL Vision-Language Model."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        # Detect if it's Qwen2-VL or original Qwen-VL
        self.is_qwen2 = "qwen2" in model_path.lower()

        if self.is_qwen2:
            self._load_qwen2_vl(model_path)
        else:
            self._load_qwen_vl(model_path)

    def _load_qwen2_vl(self, model_path: str):
        """Load Qwen2-VL model."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            print(f"Loading Qwen2-VL from {model_path}...")

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model.eval()
            print("Qwen2-VL loaded successfully.")

        except ImportError:
            raise ImportError(
                "Please install transformers and qwen-vl-utils: "
                "pip install transformers qwen-vl-utils"
            )

    def _load_qwen_vl(self, model_path: str):
        """Load original Qwen-VL model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading Qwen-VL from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            print("Qwen-VL loaded successfully.")

        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def query(self, image: Image.Image, prompt: str) -> str:
        if self.is_qwen2:
            return self._query_qwen2(image, prompt)
        else:
            return self._query_qwen(image, prompt)

    def _query_qwen2(self, image: Image.Image, prompt: str) -> str:
        """Query Qwen2-VL model."""
        # Build conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )

        # Decode - only the generated part
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response.strip()

    def _query_qwen(self, image: Image.Image, prompt: str) -> str:
        """Query original Qwen-VL model."""
        import tempfile

        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            temp_path = f.name

        try:
            query = self.tokenizer.from_list_format(
                [
                    {"image": temp_path},
                    {"text": prompt},
                ]
            )

            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=256, do_sample=False
                )

            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract response after the prompt
            if prompt in response:
                response = response.split(prompt)[-1].strip()

            return response

        finally:
            os.unlink(temp_path)


# ============================================
# InternVL VLM
# ============================================


class InternVLVLM(BaseVLM):
    """InternVL Vision-Language Model."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from transformers import AutoModel, AutoTokenizer

            print(f"Loading InternVL from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            print("InternVL loaded successfully.")

        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def query(self, image: Image.Image, prompt: str) -> str:
        # InternVL specific query format
        pixel_values = self._preprocess_image(image)

        generation_config = dict(
            max_new_tokens=256,
            do_sample=False,
        )

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer, pixel_values, prompt, generation_config
            )

        return response

    def _preprocess_image(self, image: Image.Image):
        """Preprocess image for InternVL."""
        # This will depend on the specific InternVL version
        # Placeholder implementation
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return transform(image).unsqueeze(0).to(self.device, torch.float16)


# ============================================
# VLM Factory
# ============================================


def create_vlm(config: TagExtractionConfig) -> BaseVLM:
    """Create VLM based on configuration."""
    vlm_type = config.vlm_model.lower()

    if vlm_type == "llava":
        return LLaVAVLM(config.vlm_model_path, config.vlm_device)
    elif vlm_type in ["qwen-vl", "qwen_vl", "qwenvl"]:
        return QwenVLVLM(config.vlm_model_path, config.vlm_device)
    elif vlm_type in ["internvl", "intern-vl"]:
        return InternVLVLM(config.vlm_model_path, config.vlm_device)
    else:
        raise ValueError(
            f"Unknown VLM type: {vlm_type}. " f"Supported: llava, qwen-vl, internvl"
        )


# ============================================
# Base LLM Class
# ============================================


class BaseLLM(ABC):
    """Base class for Large Language Models."""

    @abstractmethod
    def query(self, prompt: str) -> str:
        """Query the LLM with a prompt."""
        pass

    def classify_tags(
        self, tags: List[str], task_description: str, batch_size: int = 100
    ) -> Dict[str, List[str]]:
        """Classify tags as relevant or irrelevant."""
        all_relevant = []
        all_irrelevant = []
        all_reasoning = {}

        # Process in batches
        for i in tqdm(range(0, len(tags), batch_size), desc="Classifying tags"):
            batch = tags[i : i + batch_size]

            prompt = LLM_RELEVANCE_PROMPT.format(
                task_description=task_description, tag_list=", ".join(batch)
            )

            response = self.query(prompt)
            result = self._parse_response(response, batch)

            all_relevant.extend(result.get("relevant", []))
            all_irrelevant.extend(result.get("irrelevant", []))
            all_reasoning.update(result.get("reasoning", {}))

        return {
            "relevant": all_relevant,
            "irrelevant": all_irrelevant,
            "reasoning": all_reasoning,
        }

    def _parse_response(self, response: str, expected_tags: List[str]) -> Dict:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: try to parse manually
        print(f"Warning: Could not parse JSON response, using fallback parsing")
        print(f"Response: {response[:500]}...")

        # Default all to irrelevant if parsing fails
        return {
            "relevant": [],
            "irrelevant": expected_tags,
            "reasoning": {"_parse_error": "Failed to parse LLM response"},
        }


# ============================================
# OpenAI LLM
# ============================================


class OpenAILLM(BaseLLM):
    """OpenAI GPT models."""

    def __init__(self, model: str = "gpt-4", api_key: str = ""):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
            )

        try:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def query(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=4096,
        )
        return response.choices[0].message.content


# ============================================
# Ollama LLM (Local)
# ============================================


class OllamaLLM(BaseLLM):
    """Ollama local LLM."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

        try:
            import ollama

            self.client = ollama
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")

    def query(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0},
        )
        return response["message"]["content"]


# ============================================
# Transformers LLM (Local HuggingFace)
# ============================================


class TransformersLLM(BaseLLM):
    """Local LLM using HuggingFace Transformers."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading LLM from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=self.device
            )
            self.model.eval()
            print("LLM loaded successfully.")

        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def query(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=4096, temperature=0.0, do_sample=False
            )

        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response


# ============================================
# LLM Factory
# ============================================


def create_llm(config: TagExtractionConfig) -> BaseLLM:
    """Create LLM based on configuration."""
    llm_type = config.llm_model.lower()

    if llm_type in ["gpt-5.4", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"]:
        return OpenAILLM(model=llm_type, api_key=config.llm_api_key)
    elif llm_type in ["ollama", "llama3", "llama2", "mistral"]:
        model_name = llm_type if llm_type != "ollama" else "llama3"
        return OllamaLLM(
            model=model_name, base_url=config.llm_base_url or "http://localhost:11434"
        )
    elif llm_type == "local" or config.llm_model_path:
        return TransformersLLM(config.llm_model_path, config.vlm_device)
    else:
        raise ValueError(
            f"Unknown LLM type: {llm_type}. "
            f"Supported: gpt-4, gpt-3.5-turbo, ollama, llama3, local"
        )


# ============================================
# Main Pipeline
# ============================================


class TagExtractionPipeline:
    """
    Main pipeline for comprehensive tag extraction and relevance classification.
    """

    def __init__(self, config: TagExtractionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)

        # Create output directories
        self.stage1_dir = self.output_dir / "stage1_extraction"
        self.stage2_dir = self.output_dir / "stage2_vocabulary"
        self.stage3_dir = self.output_dir / "stage3_relevance"
        self.final_dir = self.output_dir / "final"

        for d in [self.stage1_dir, self.stage2_dir, self.stage3_dir, self.final_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Models (lazy loaded)
        self._vlm = None
        self._llm = None

    @property
    def vlm(self) -> BaseVLM:
        if self._vlm is None:
            self._vlm = create_vlm(self.config)
        return self._vlm

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm(self.config)
        return self._llm

    def run(
        self,
        image_paths: List[str],
        image_indices: List[int],
        image_targets: List[int],
        task_description: str,
        resume_from_stage: int = 1,
    ):
        """
        Run the complete pipeline.

        Args:
            image_paths: List of paths to images
            image_indices: List of image indices (for output CSV)
            image_targets: List of target labels
            task_description: Description of the classification task
            resume_from_stage: Stage to resume from (1-4)
        """
        self.config.task_description = task_description

        print(f"\n{'='*60}")
        print("Comprehensive Tag Extraction Pipeline")
        print(f"{'='*60}")
        print(f"Task: {task_description}")
        print(f"Images: {len(image_paths)}")
        print(f"Output: {self.output_dir}")
        print(f"Resume from: Stage {resume_from_stage}")

        # Stage 1: Extract tags from images
        if resume_from_stage <= 1:
            image_tags = self._stage1_extract(image_paths, image_indices)
        else:
            image_tags = self._load_stage1()

        # Stage 2: Aggregate vocabulary
        if resume_from_stage <= 2:
            vocabulary = self._stage2_aggregate(image_tags)
        else:
            vocabulary = self._load_stage2()

        # Stage 3: Classify relevance
        if resume_from_stage <= 3:
            relevance = self._stage3_classify(vocabulary, task_description)
        else:
            relevance = self._load_stage3()

        # Optional: Human review
        if self.config.enable_human_review:
            relevance = self._human_review(relevance, vocabulary)

        # Stage 4: Apply mapping and generate output
        self._stage4_apply(image_tags, image_indices, image_targets, relevance)

        print(f"\n{'='*60}")
        print("Pipeline Complete!")
        print(f"{'='*60}")
        print(f"Output CSV: {self.final_dir / 'train_tags.csv'}")

    def _stage1_extract(
        self, image_paths: List[str], image_indices: List[int]
    ) -> Dict[int, List[str]]:
        """Stage 1: Extract tags from all images with incremental saving."""
        print(f"\n{'='*60}")
        print("Stage 1: Extracting Tags from Images")
        print(f"{'='*60}")

        output_path = self.stage1_dir / "image_tags.json"
        error_path = self.stage1_dir / "extraction_errors.json"

        # Load existing progress if available
        image_tags = {}
        errors = []

        if output_path.exists():
            print(f"  Found existing progress, resuming...")
            with open(output_path, "r") as f:
                existing = json.load(f)
                image_tags = {int(k): v for k, v in existing.items()}
            print(f"  Loaded {len(image_tags)} already processed images")

        if error_path.exists():
            with open(error_path, "r") as f:
                errors = json.load(f)

        # Filter out already processed images
        remaining_indices = []
        remaining_paths = []
        for idx, path in zip(image_indices, image_paths):
            if idx not in image_tags:
                remaining_indices.append(idx)
                remaining_paths.append(path)

        print(f"  Remaining images to process: {len(remaining_indices)}")

        if len(remaining_indices) == 0:
            print("  All images already processed!")
            return image_tags

        # Process remaining images with incremental saving
        save_interval = 50  # Save every N images

        for i, (idx, path) in enumerate(
            tqdm(
                zip(remaining_indices, remaining_paths),
                total=len(remaining_paths),
                desc="Extracting tags",
            )
        ):
            try:
                image = Image.open(path).convert("RGB")
                result = self.vlm.extract_tags(image)
                image_tags[idx] = result["all_tags"]
            except Exception as e:
                errors.append((idx, path, str(e)))
                image_tags[idx] = []

            # Incremental save every N images
            if (i + 1) % save_interval == 0:
                with open(output_path, "w") as f:
                    json.dump(image_tags, f, indent=2)
                if errors:
                    with open(error_path, "w") as f:
                        json.dump(errors, f, indent=2)

        # Final save
        with open(output_path, "w") as f:
            json.dump(image_tags, f, indent=2)

        # Save errors if any
        if errors:
            with open(error_path, "w") as f:
                json.dump(errors, f, indent=2)
            print(
                f"  Warnings: {len(errors)} images failed (see extraction_errors.json)"
            )

        print(f"  Extracted tags from {len(image_tags)} images")
        print(f"  Saved to: {output_path}")

        return image_tags

    def _load_stage1(self) -> Dict[int, List[str]]:
        """Load Stage 1 results."""
        path = self.stage1_dir / "image_tags.json"
        print(f"Loading Stage 1 from: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        # Convert string keys back to int
        return {int(k): v for k, v in data.items()}

    def _stage2_aggregate(self, image_tags: Dict[int, List[str]]) -> Dict[str, int]:
        """Stage 2: Aggregate vocabulary."""
        print(f"\n{'='*60}")
        print("Stage 2: Aggregating Vocabulary")
        print(f"{'='*60}")

        # Count tag frequencies
        tag_counts = defaultdict(int)
        for tags in image_tags.values():
            for tag in tags:
                # Normalize: lowercase, strip
                tag = tag.lower().strip()
                if tag:
                    tag_counts[tag] += 1

        # Filter by minimum frequency
        min_freq = self.config.min_tag_frequency
        vocabulary = {
            tag: count
            for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])
            if count >= min_freq
        }

        # Save vocabulary
        output_path = self.stage2_dir / "tag_vocabulary.json"
        with open(output_path, "w") as f:
            json.dump(vocabulary, f, indent=2)

        # Save statistics
        stats = {
            "total_tags_extracted": sum(tag_counts.values()),
            "unique_tags_all": len(tag_counts),
            "unique_tags_filtered": len(vocabulary),
            "min_frequency_threshold": min_freq,
            "tags_removed_by_filter": len(tag_counts) - len(vocabulary),
        }
        stats_path = self.stage2_dir / "tag_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  Total unique tags: {len(tag_counts)}")
        print(f"  After filtering (freq >= {min_freq}): {len(vocabulary)}")
        print(f"  Saved to: {output_path}")

        return vocabulary

    def _load_stage2(self) -> Dict[str, int]:
        """Load Stage 2 results."""
        path = self.stage2_dir / "tag_vocabulary.json"
        print(f"Loading Stage 2 from: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _stage3_classify(
        self, vocabulary: Dict[str, int], task_description: str
    ) -> Dict[str, Any]:
        """Stage 3: Classify tags as relevant/irrelevant."""
        print(f"\n{'='*60}")
        print("Stage 3: Classifying Tag Relevance")
        print(f"{'='*60}")
        print(f"  Task: {task_description}")
        print(f"  Tags to classify: {len(vocabulary)}")

        tags = list(vocabulary.keys())

        result = self.llm.classify_tags(
            tags=tags,
            task_description=task_description,
            batch_size=self.config.llm_tag_batch_size,
        )

        # Add metadata
        relevance = {
            "task": task_description,
            "statistics": {
                "total_tags": len(tags),
                "relevant_count": len(result["relevant"]),
                "irrelevant_count": len(result["irrelevant"]),
            },
            "relevant": result["relevant"],
            "irrelevant": result["irrelevant"],
            "reasoning": result["reasoning"],
        }

        # Check for unclassified tags
        classified = set(result["relevant"]) | set(result["irrelevant"])
        unclassified = set(tags) - classified
        if unclassified:
            print(
                f"  Warning: {len(unclassified)} tags not classified, marking as irrelevant"
            )
            relevance["irrelevant"].extend(list(unclassified))
            relevance["statistics"]["irrelevant_count"] += len(unclassified)

        # Save results
        output_path = self.stage3_dir / "tag_relevance_auto.json"
        with open(output_path, "w") as f:
            json.dump(relevance, f, indent=2)

        print(f"  Relevant tags: {relevance['statistics']['relevant_count']}")
        print(f"  Irrelevant tags: {relevance['statistics']['irrelevant_count']}")
        print(f"  Saved to: {output_path}")

        return relevance

    def _load_stage3(self) -> Dict[str, Any]:
        """Load Stage 3 results."""
        # Try final first, then auto
        final_path = self.stage3_dir / "tag_relevance_final.json"
        auto_path = self.stage3_dir / "tag_relevance_auto.json"

        if final_path.exists():
            path = final_path
        else:
            path = auto_path

        print(f"Loading Stage 3 from: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _human_review(
        self, relevance: Dict[str, Any], vocabulary: Dict[str, int]
    ) -> Dict[str, Any]:
        """Prepare and wait for human review."""
        print(f"\n{'='*60}")
        print("Human Review")
        print(f"{'='*60}")

        # Create review file with instructions
        review_data = {
            "instructions": (
                "Review the tag classifications below.\n"
                "Move tags between 'relevant' and 'irrelevant' as needed.\n"
                "Save this file and re-run the pipeline with resume_from_stage=4.\n"
                "Or rename this file to 'tag_relevance_final.json' to apply changes."
            ),
            "task": relevance["task"],
            "statistics": relevance["statistics"],
            "relevant": {
                tag: relevance["reasoning"].get(tag, "")
                for tag in relevance["relevant"]
            },
            "irrelevant": {
                tag: relevance["reasoning"].get(tag, "")
                for tag in relevance["irrelevant"]
            },
        }

        review_path = self.stage3_dir / "tag_relevance_review.json"
        with open(review_path, "w") as f:
            json.dump(review_data, f, indent=2)

        print(f"  Review file created: {review_path}")
        print(f"  Edit this file to adjust classifications.")
        print(f"  Then rename to 'tag_relevance_final.json' or re-run with stage=4")

        # Check if final exists
        final_path = self.stage3_dir / "tag_relevance_final.json"
        if final_path.exists():
            print(f"  Found final file, using: {final_path}")
            with open(final_path, "r") as f:
                final_data = json.load(f)

            # Convert reviewed format back to pipeline format
            return {
                "task": final_data["task"],
                "statistics": final_data.get("statistics", relevance["statistics"]),
                "relevant": (
                    list(final_data["relevant"].keys())
                    if isinstance(final_data["relevant"], dict)
                    else final_data["relevant"]
                ),
                "irrelevant": (
                    list(final_data["irrelevant"].keys())
                    if isinstance(final_data["irrelevant"], dict)
                    else final_data["irrelevant"]
                ),
                "reasoning": (
                    {
                        **final_data.get("relevant", {}),
                        **final_data.get("irrelevant", {}),
                    }
                    if isinstance(final_data.get("relevant"), dict)
                    else relevance["reasoning"]
                ),
            }

        return relevance

    def _stage4_apply(
        self,
        image_tags: Dict[int, List[str]],
        image_indices: List[int],
        image_targets: List[int],
        relevance: Dict[str, Any],
    ):
        """Stage 4: Apply mapping and generate output CSV."""
        print(f"\n{'='*60}")
        print("Stage 4: Generating Output CSV")
        print(f"{'='*60}")

        # Create lookup sets
        relevant_set = set(relevance["relevant"])
        irrelevant_set = set(relevance["irrelevant"])

        # Generate CSV
        output_path = self.final_dir / "train_tags.csv"
        separator = " | "

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["index", "target", "tags", "relevant_tags", "irrelevant_tags"]
            )

            for idx, target in zip(image_indices, image_targets):
                tags = image_tags.get(idx, [])

                # Normalize and split
                tags_normalized = [t.lower().strip() for t in tags]

                relevant = [t for t in tags_normalized if t in relevant_set]
                irrelevant = [t for t in tags_normalized if t in irrelevant_set]

                writer.writerow(
                    [
                        idx,
                        target.item(),
                        separator.join(tags_normalized),
                        separator.join(relevant),
                        separator.join(irrelevant),
                    ]
                )

        # Save final relevance mapping
        final_path = self.stage3_dir / "tag_relevance_final.json"
        with open(final_path, "w") as f:
            json.dump(relevance, f, indent=2)

        print(f"  Output CSV: {output_path}")
        print(f"  Total rows: {len(image_indices)}")


# ============================================
# Convenience function
# ============================================


def extract_tags_for_dataset(
    image_paths: List[str],
    image_indices: List[int],
    image_targets: List[int],
    task_description: str,
    output_dir: str = "./tag_extraction_output",
    vlm_model: str = "llava",
    vlm_model_path: str = "llava-hf/llava-1.5-7b-hf",
    llm_model: str = "ollama",
    llm_model_path: str = "",
    enable_human_review: bool = False,
    min_tag_frequency: int = 5,
    resume_from_stage: int = 1,
):
    """
    Convenience function to run the complete tag extraction pipeline.

    Args:
        image_paths: List of image file paths
        image_indices: List of image indices
        image_targets: List of target labels
        task_description: Description of the classification task
                         (e.g., "car type classification", "bird species classification")
        output_dir: Directory for output files
        vlm_model: VLM to use ("llava", "qwen-vl", "internvl")
        vlm_model_path: Path or HuggingFace model ID for VLM
        llm_model: LLM to use ("gpt-4", "ollama", "llama3", "local")
        llm_model_path: Path for local LLM
        enable_human_review: Whether to enable human review step
        min_tag_frequency: Minimum tag frequency threshold
        resume_from_stage: Stage to resume from (1-4)

    Returns:
        Path to output CSV file
    """
    config = TagExtractionConfig(
        vlm_model=vlm_model,
        vlm_model_path=vlm_model_path,
        llm_model=llm_model,
        llm_model_path=llm_model_path,
        enable_human_review=enable_human_review,
        min_tag_frequency=min_tag_frequency,
        output_dir=output_dir,
    )

    pipeline = TagExtractionPipeline(config)
    pipeline.run(
        image_paths=image_paths,
        image_indices=image_indices,
        image_targets=image_targets,
        task_description=task_description,
        resume_from_stage=resume_from_stage,
    )

    return str(pipeline.final_dir / "train_tags.csv")
