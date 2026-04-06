"""
Vision-Language Model Encoders for VB-Mitigator.

This module provides a unified interface for various VLM encoders:
- OpenCLIP (CLIP variants from OpenAI and LAION)
- SigLIP (Google's Sigmoid Loss for Language Image Pre-Training)
- Perception Encoder (Meta's state-of-the-art vision encoder)

All encoders support:
1. Image encoding (frozen backbone)
2. Text encoding (for zero-shot classification)
3. Zero-shot classification via text-image similarity
4. Classification head training (linear or MLP)

Each encoder can be used with SAE analysis and the neuron classifier.
"""

import os
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# Dataset-specific class names for zero-shot
# ============================================

DATASET_CLASS_NAMES = {
    # UTKFace: Gender classification (Male/Female)
    "utkface": {
        "default": ["a photo of a male person", "a photo of a female person"],
        "short": ["male", "female"],
        "detailed": ["a photograph of a man", "a photograph of a woman"],
    },
    # Waterbirds: Bird species (not just waterbird/landbird)
    "waterbirds": {
        "default": ["a photo of a landbird", "a photo of a waterbird"],
        "species": [
            # Landbirds (class 0)
            "a photo of a warbler, a small songbird",
            # Waterbirds (class 1)
            "a photo of an albatross, a seabird with long wings",
        ],
        "detailed": [
            "a photo of a landbird, such as a warbler, sparrow, or finch",
            "a photo of a seabird, such as an albatross, pelican, or gull",
        ],
    },
    # CelebA: Various binary attributes
    "celeba": {
        "default": [
            "a photo of a person without the attribute",
            "a photo of a person with the attribute",
        ],
        "blonde": [
            "a photo of a person with non-blond hair",
            "a photo of a person with blond hair",
        ],
        "male": ["a photo of a female person", "a photo of a male person"],
        "young": ["a photo of an older person", "a photo of a young person"],
        "smiling": [
            "a photo of a person with a neutral expression",
            "a photo of a smiling person",
        ],
    },
    # UrbanCars: Car types (not just urban/country)
    "urbancars": {
        "default": [
            "a photo of a car in an urban environment",
            "a photo of a car in a rural/country environment",
        ],
        # "car_types": [
        #     "a photograph of a AM General Hummer SUV 2000, Aston Martin V8 Vantage Convertible 2012, BMW X5 SUV 2007, BMW X6 SUV 2012, BMW X3 SUV 2012, Buick Rainier SUV 2007, Buick Enclave SUV 2012, Cadillac SRX SUV 2012, Cadillac Escalade EXT Crew Cab 2007, Chevrolet Silverado 1500 Hybrid Crew Cab 2012, Chevrolet Traverse SUV 2012, Chevrolet HHR SS 2010, Chevrolet Tahoe Hybrid SUV 2012, Chevrolet Express Cargo Van 2007, Chevrolet Avalanche Crew Cab 2012, Chevrolet TrailBlazer SS 2009, Chevrolet Silverado 2500HD Regular Cab 2012, Chevrolet Silverado 1500 Classic Extended Cab 2007, Chevrolet Express Van 2007, Chevrolet Silverado 1500 Extended Cab 2012, Chevrolet Silverado 1500 Regular Cab 2012, Chrysler Aspen SUV 2009, Chrysler Town and Country Minivan 2012, Dodge Caravan Minivan 1997, Dodge Ram Pickup 3500 Crew Cab 2010, Dodge Ram Pickup 3500 Quad Cab 2009, Dodge Sprinter Cargo Van 2009, Dodge Journey SUV 2012, Dodge Dakota Crew Cab 2010, Dodge Dakota Club Cab 2007, Dodge Durango SUV 2012, Dodge Durango SUV 2007, Ford F-450 Super Duty Crew Cab 2012, Ford Freestar Minivan 2007, Ford Expedition EL SUV 2009, Ford Edge SUV 2012, Ford Ranger SuperCab 2011, Ford F-150 Regular Cab 2012, Ford F-150 Regular Cab 2007, Ford E-Series Wagon Van 2012, GMC Terrain SUV 2012, GMC Savana Van 2012, GMC Yukon Hybrid SUV 2012, GMC Acadia SUV 2012, GMC Canyon Extended Cab 2012, HUMMER H3T Crew Cab 2010, HUMMER H2 SUT Crew Cab 2009, Honda Odyssey Minivan 2012, Honda Odyssey Minivan 2007, Hyundai Santa Fe SUV 2012, Hyundai Tucson SUV 2012, Hyundai Veracruz SUV 2012, Infiniti QX56 SUV 2011, Isuzu Ascender SUV 2008, Jeep Patriot SUV 2012, Jeep Wrangler SUV 2012, Jeep Liberty SUV 2012, Jeep Grand Cherokee SUV 2012, Jeep Compass SUV 2012, Land Rover Range Rover SUV 2012, Land Rover LR2 SUV 2012, Mazda Tribute SUV 2011, Mercedes-Benz Sprinter Van 2012, Nissan NV Passenger Van 2012, Ram C/V Cargo Van Minivan 2012, Toyota Sequoia SUV 2012, Toyota 4Runner SUV 2012, Volvo XC90 SUV 2007",
        #     "a photograph of a Acura RL Sedan 2012, Acura TL Sedan 2012, Acura TL Type-S 2008, Acura TSX Sedan 2012, Acura Integra Type R 2001, Acura ZDX Hatchback 2012, Aston Martin V8 Vantage Coupe 2012, Aston Martin Virage Convertible 2012, Aston Martin Virage Coupe 2012, Audi RS 4 Convertible 2008, Audi A5 Coupe 2012, Audi TTS Coupe 2012, Audi R8 Coupe 2012, Audi V8 Sedan 1994, Audi 100 Sedan 1994, Audi 100 Wagon 1994, Audi TT Hatchback 2011, Audi S6 Sedan 2011, Audi S5 Convertible 2012, Audi S5 Coupe 2012, Audi S4 Sedan 2012, Audi S4 Sedan 2007, Audi TT RS Coupe 2012, BMW ActiveHybrid 5 Sedan 2012, BMW 1 Series Convertible 2012, BMW 1 Series Coupe 2012, BMW 3 Series Sedan 2012, BMW 3 Series Wagon 2012, BMW 6 Series Convertible 2007, BMW M3 Coupe 2012, BMW M5 Sedan 2010, BMW M6 Convertible 2010, BMW Z4 Convertible 2012, Bentley Continental Supersports Conv. Convertible 2012, Bentley Arnage Sedan 2009, Bentley Mulsanne Sedan 2011, Bentley Continental GT Coupe 2012, Bentley Continental GT Coupe 2007, Bentley Continental Flying Spur Sedan 2007, Bugatti Veyron 16.4 Convertible 2009, Bugatti Veyron 16.4 Coupe 2009, Buick Regal GS 2012, Buick Verano Sedan 2012, Cadillac CTS-V Sedan 2012, Chevrolet Corvette Convertible 2012, Chevrolet Corvette ZR1 2012, Chevrolet Corvette Ron Fellows Edition Z06 2007, Chevrolet Camaro Convertible 2012, Chevrolet Impala Sedan 2007, Chevrolet Sonic Sedan 2012, Chevrolet Cobalt SS 2010, Chevrolet Malibu Hybrid Sedan 2010, Chevrolet Monte Carlo Coupe 2007, Chevrolet Malibu Sedan 2007, Chrysler Sebring Convertible 2010, Chrysler 300 SRT-8 2010, Chrysler Crossfire Convertible 2008, Chrysler PT Cruiser Convertible 2008, Daewoo Nubira Wagon 2002, Dodge Caliber Wagon 2012, Dodge Caliber Wagon 2007, Dodge Magnum Wagon 2008, Dodge Challenger SRT8 2011, Dodge Charger Sedan 2012, Dodge Charger SRT-8 2009, Eagle Talon Hatchback 1998, FIAT 500 Abarth 2012, FIAT 500 Convertible 2012, Ferrari FF Coupe 2012, Ferrari California Convertible 2012, Ferrari 458 Italia Convertible 2012, Ferrari 458 Italia Coupe 2012, Fisker Karma Sedan 2012, Ford Mustang Convertible 2007, Ford GT Coupe 2006, Ford Focus Sedan 2007, Ford Fiesta Sedan 2012, Geo Metro Convertible 1993, Honda Accord Coupe 2012, Honda Accord Sedan 2012, Hyundai Veloster Hatchback 2012, Hyundai Sonata Hybrid Sedan 2012, Hyundai Elantra Sedan 2007, Hyundai Accent Sedan 2012, Hyundai Genesis Sedan 2012, Hyundai Sonata Sedan 2012, Hyundai Elantra Touring Hatchback 2012, Hyundai Azera Sedan 2012, Infiniti G Coupe IPL2012, Jaguar XK XKR 2012, Lamborghini Reventon Coupe 2008, Lamborghini Aventador Coupe 2012, Lamborghini Gallardo LP 570-4 Superleggera 2012, Lamborghini Diablo Coupe 2001, Lincoln Town Car Sedan 2011, MINI Cooper Roadster Convertible 2012, Maybach Landaulet Convertible 2012, McLaren MP4-12C Coupe 2012, Mercedes-Benz 300-Class Convertible 1993, Mercedes-Benz C-Class Sedan 2012, Mercedes-Benz SL-Class Coupe 2009, Mercedes-Benz E-Class Sedan 2012, Mercedes-Benz S-Class Sedan 2012, Mitsubishi Lancer Sedan 2012, Nissan Leaf Hatchback 2012, Nissan Juke Hatchback 2012, Nissan 240SX Coupe 1998, Plymouth Neon Coupe 1999, Porsche Panamera Sedan 2012, Rolls-Royce Phantom Drophead Coupe Convertible 2012, Rolls-Royce Ghost Sedan 2012, Rolls-Royce Phantom Sedan 2012, Scion xD Hatchback 2012, Spyker C8 Convertible 2009, Spyker C8 Coupe 2009, Suzuki Aerio Sedan 2007, Suzuki Kizashi Sedan 2012, Suzuki SX4 Hatchback 2012, Suzuki SX4 Sedan 2012, Tesla Model S Sedan 2012, Toyota Camry Sedan 2012, Toyota Corolla Sedan 2012, Volkswagen Golf Hatchback 2012, Volkswagen Golf Hatchback 1991, Volkswagen Beetle Hatchback 2012, Volvo C30 Hatchback 2012, Volvo 240 Sedan 1993, smart fortwo Convertible 2012",
        # ],
        "car_types": [
            "a photograph of a compact, sports, sedan car",
            "a photograph of a truck, jeep, pickup car",
        ],
        "detailed": [
            "a photograph of a compact car, sports car, or luxury vehicle",
            "a photograph of a truck, SUV, or rugged vehicle",
        ],
    },
    # ImageNet (for general testing)
    "imagenet": {
        "template": "a photo of a {}",  # Will be filled with class name
    },
    # CIFAR-10
    "cifar10": {
        "default": [
            "a photo of an airplane",
            "a photo of an automobile",
            "a photo of a bird",
            "a photo of a cat",
            "a photo of a deer",
            "a photo of a dog",
            "a photo of a frog",
            "a photo of a horse",
            "a photo of a ship",
            "a photo of a truck",
        ],
    },
}


def get_class_names(dataset_name: str, variant: str = "default") -> List[str]:
    """
    Get class names for zero-shot classification.

    Args:
        dataset_name: Name of dataset (e.g., "utkface", "waterbirds")
        variant: Which variant of class names to use (e.g., "default", "species", "detailed")

    Returns:
        List of text prompts for each class
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_CLASS_NAMES:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CLASS_NAMES.keys())}"
        )

    class_config = DATASET_CLASS_NAMES[dataset_name]

    if variant in class_config:
        return class_config[variant]
    elif "default" in class_config:
        return class_config["default"]
    else:
        raise ValueError(f"Unknown variant '{variant}' for dataset '{dataset_name}'")


# ============================================
# Base VLM Encoder Class
# ============================================


class BaseVLMEncoder(nn.Module):
    """
    Base class for Vision-Language Model encoders.

    All subclasses should implement:
    - encode_image(images) -> image_features
    - encode_text(texts) -> text_features
    - get_image_size() -> (height, width)
    - get_embed_dim() -> int
    """

    def __init__(self):
        super().__init__()
        self._image_size = None
        self._embed_dim = None

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        raise NotImplementedError

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts to feature vectors."""
        raise NotImplementedError

    def zero_shot_classify(
        self, images: torch.Tensor, class_names: List[str], normalize: bool = True
    ) -> torch.Tensor:
        """
        Zero-shot classification via image-text similarity.

        Args:
            images: Image tensor (B, C, H, W)
            class_names: List of text prompts for each class
            normalize: Whether to L2 normalize features

        Returns:
            Similarity scores (B, num_classes)
        """
        # Encode images
        image_features = self.encode_image(images)
        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        # Encode text
        text_features = self.encode_text(class_names)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)

        # Compute similarity
        similarity = image_features @ text_features.T

        return similarity


# ============================================
# OpenCLIP Encoder
# ============================================


class OpenCLIPEncoder(BaseVLMEncoder):
    """
    OpenCLIP encoder supporting various CLIP models.

    Supported architectures:
    - ViT-B/32, ViT-B/16, ViT-L/14, ViT-H/14, ViT-G/14
    - RN50, RN101, RN50x4, RN50x16, RN50x64

    Pretrained sources:
    - "openai": Original OpenAI CLIP weights
    - "laion2b_s34b_b79k": LAION-2B trained weights
    - "laion400m_e32": LAION-400M trained weights
    """

    MODEL_CONFIGS = {
        "ViT-B-32": (512, 224),
        "ViT-B-16": (512, 224),
        "ViT-L-14": (768, 224),
        "ViT-L-14-336": (768, 336),
        "ViT-H-14": (1024, 224),
        "ViT-G-14": (1280, 224),
        "RN50": (1024, 224),
        "RN101": (512, 224),
        "RN50x4": (640, 288),
        "RN50x16": (768, 384),
        "RN50x64": (1024, 448),
        # SigLIP models
        "ViT-SO400M-14-SigLIP": (1152, 224),
        "ViT-SO400M-14-SigLIP-384": (1152, 384),
        "ViT-B-16-SigLIP": (768, 224),
        "ViT-B-16-SigLIP-256": (768, 256),
        "ViT-B-16-SigLIP-384": (768, 384),
        "ViT-B-16-SigLIP-512": (768, 512),
        "ViT-L-16-SigLIP-256": (1024, 256),
        "ViT-L-16-SigLIP-384": (1024, 384),
    }

    def __init__(
        self,
        arch: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        super().__init__()

        try:
            import open_clip
        except ImportError:
            raise ImportError("Please install open_clip: pip install open_clip_torch")

        # Load model and tokenizer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(arch)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Get embed_dim and image_size from model or config
        if arch in self.MODEL_CONFIGS:
            self._embed_dim, img_size = self.MODEL_CONFIGS[arch]
            self._image_size = (img_size, img_size)
        else:
            # Query from loaded model
            self._embed_dim = self.model.visual.output_dim
            # Try to get image size from model config or preprocess
            if hasattr(self.model.visual, "image_size"):
                img_size = self.model.visual.image_size
                if isinstance(img_size, tuple):
                    self._image_size = img_size
                else:
                    self._image_size = (img_size, img_size)
            else:
                self._image_size = (224, 224)  # Default fallback

        self.device = device
        self.arch = arch
        self.pretrained = pretrained

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to CLIP feature vectors."""
        with torch.no_grad():
            features = self.model.encode_image(images)
        return features

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text to CLIP feature vectors."""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features

    def get_transform(self):
        """Get the image preprocessing transform."""
        return self.preprocess


# ============================================
# SigLIP Encoder
# ============================================


class SigLIPEncoder(BaseVLMEncoder):
    """
    SigLIP encoder (Google's Sigmoid Loss for Language Image Pre-Training).

    Uses HuggingFace transformers for loading.

    Supported models:
    - siglip-base-patch16-224
    - siglip-base-patch16-384
    - siglip-large-patch16-256
    - siglip-large-patch16-384
    - siglip-so400m-patch14-384
    """

    MODEL_CONFIGS = {
        "siglip-base-patch16-224": (768, 224),
        "siglip-base-patch16-384": (768, 384),
        "siglip-large-patch16-256": (1024, 256),
        "siglip-large-patch16-384": (1024, 384),
        "siglip-so400m-patch14-384": (1152, 384),
    }

    def __init__(
        self,
        model_name: str = "siglip-so400m-patch14-384",
        device: str = "cuda",
        pretrained: str = "none",
    ):
        super().__init__()

        try:
            from transformers import AutoProcessor, SiglipModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(self.MODEL_CONFIGS.keys())}"
            )

        self._embed_dim, img_size = self.MODEL_CONFIGS[model_name]
        self._image_size = (img_size, img_size)

        # Load full SigLIP model (vision + text)
        hf_name = f"google/{model_name}"
        self.model = SiglipModel.from_pretrained(hf_name).to(device)
        self.processor = AutoProcessor.from_pretrained(hf_name)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.model_name = model_name

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to SigLIP feature vectors."""
        with torch.no_grad():
            outputs = self.model.get_image_features(pixel_values=images)
        return outputs

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text to SigLIP feature vectors."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {
            k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"
        }

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs

    def get_processor(self):
        """Get the processor for image/text preprocessing."""
        return self.processor


# ============================================
# Perception Encoder
# ============================================


class PerceptionEncoder(BaseVLMEncoder):
    """
    Meta's Perception Encoder (PE) - state-of-the-art vision encoder.

    Can be loaded via:
    1. Official perception_models repo (recommended)
    2. timm (image encoder only)
    3. open_clip (PE-Core models)

    Supported models:
    - PE-Core-B16-224
    - PE-Core-L14-336
    - PE-Core-G14-448
    - PE-Core-S16-384
    - PE-Core-T16-384
    """

    MODEL_CONFIGS = {
        "PE-Core-B16-224": (768, 224),
        "PE-Core-L14-336": (1024, 336),
        "PE-Core-G14-448": (1408, 448),
        "PE-Core-S16-384": (768, 384),
        "PE-Core-T16-384": (512, 384),
    }

    def __init__(
        self,
        model_name: str = "PE-Core-L14-336",
        device: str = "cuda",
        backend: str = "native",  # "open_clip", "timm", or "native"
        pretrained: str = "none",
    ):
        super().__init__()

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(self.MODEL_CONFIGS.keys())}"
            )

        self._embed_dim, img_size = self.MODEL_CONFIGS[model_name]
        self._image_size = (img_size, img_size)

        self.model_name = model_name
        self.device = device
        self.backend = backend

        if backend == "open_clip":
            self._init_open_clip(model_name, device)
        elif backend == "timm":
            self._init_timm(model_name, device)
        elif backend == "native":
            self._init_native(model_name, device)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_open_clip(self, model_name: str, device: str):
        """Initialize via open_clip (supports text encoding)."""
        try:
            import open_clip
        except ImportError:
            raise ImportError("Please install open_clip: pip install open_clip_torch")

        # Map PE model names to open_clip format
        # PE models are available in open_clip 3.0+
        oc_name = model_name.replace("-", "/").replace("PE/Core/", "PE-Core-")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="metaclip_fullcc", device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _init_timm(self, model_name: str, device: str):
        """Initialize via timm (image encoder only)."""
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        # Map to timm model names
        timm_name_map = {
            "PE-Core-B16-224": "vit_pe_core_base_patch16_224.fb",
            "PE-Core-L14-336": "vit_pe_core_large_patch14_336.fb",
            "PE-Core-G14-448": "vit_pe_core_giant_patch14_448.fb",
        }

        timm_name = timm_name_map.get(model_name, model_name)

        self.model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        self.model = self.model.to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        # Get transform
        config = resolve_data_config(self.model.pretrained_cfg)
        self.preprocess = create_transform(**config)

        # No text encoder in timm-only mode
        self.tokenizer = None
        self._has_text = False

    def _init_native(self, model_name: str, device: str):
        """Initialize via official perception_models repo."""
        try:
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as transforms
        except ImportError:
            raise ImportError(
                "Please install perception_models: "
                "git clone https://github.com/facebookresearch/perception_models && "
                "pip install -e perception_models"
            )

        self.model = pe.CLIP.from_config(model_name, pretrained=True)
        self.model = self.model.to(device)
        self.model.eval()

        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

        for param in self.model.parameters():
            param.requires_grad = False

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to PE feature vectors."""
        with torch.no_grad():
            if self.backend == "timm":
                features = self.model(images)
            else:
                features = self.model.encode_image(images)
        return features

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text to PE feature vectors."""
        if self.backend == "timm":
            raise RuntimeError(
                "Text encoding not available with timm backend. "
                "Use 'open_clip' or 'native' backend for text encoding."
            )

        if self.backend == "native":
            tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
        else:  # open_clip
            tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)

        return features


# ============================================
# Unified VLM Factory
# ============================================


def create_vlm_encoder(
    encoder_type: str, model_name: str, device: str = "cuda", **kwargs
) -> BaseVLMEncoder:
    """
    Factory function to create VLM encoders.

    Args:
        encoder_type: Type of encoder ("openclip", "siglip", "perception_encoder")
        model_name: Model name/architecture
        device: Device to load model on
        **kwargs: Additional arguments for specific encoders

    Returns:
        VLM encoder instance

    Examples:
        # OpenCLIP
        encoder = create_vlm_encoder("openclip", "ViT-B-32", pretrained="openai")

        # SigLIP
        encoder = create_vlm_encoder("siglip", "siglip-so400m-patch14-384")

        # Perception Encoder
        encoder = create_vlm_encoder("perception_encoder", "PE-Core-L14-336", backend="open_clip")
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "openclip":
        return OpenCLIPEncoder(arch=model_name, device=device, **kwargs)
    elif encoder_type == "siglip":
        return SigLIPEncoder(model_name=model_name, device=device, **kwargs)
    elif encoder_type in ["perception_encoder", "pe"]:
        return PerceptionEncoder(model_name=model_name, device=device, **kwargs)
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available: openclip, siglip, perception_encoder"
        )


# ============================================
# VLM Encoder with Classification Head
# ============================================


class VLMEncoderWithHead(nn.Module):
    """
    VLM encoder with trainable classification head.

    Supports:
    - Linear head
    - MLP head
    - Zero-shot classification (no head)
    """

    def __init__(
        self,
        encoder: BaseVLMEncoder,
        num_classes: int,
        head_type: str = "linear",  # "linear", "mlp", or "zero_shot"
        hidden_dim: int = 512,
        dropout: float = 0.1,
        class_names: Optional[List[str]] = None,  # For zero-shot
    ):
        super().__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.head_type = head_type
        self.class_names = class_names

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Create classification head
        embed_dim = encoder.embed_dim

        if head_type == "linear":
            self.head = nn.Linear(embed_dim, num_classes)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        elif head_type == "zero_shot":
            self.head = None
            if class_names is None:
                raise ValueError("class_names required for zero_shot head")
            # Pre-compute text features
            with torch.no_grad():
                text_features = encoder.encode_text(class_names)
                self.register_buffer(
                    "text_features", F.normalize(text_features, dim=-1)
                )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    @property
    def embed_size(self):
        return self.encoder.embed_dim

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)
            return_features: If True, also return features

        Returns:
            logits: Classification logits (B, num_classes)
            features (optional): Image features (B, embed_dim)
        """
        # Encode images
        with torch.no_grad():
            features = self.encoder.encode_image(x)
            features = F.normalize(features, dim=-1)

        # Classify
        if self.head_type == "zero_shot":
            logits = features @ self.text_features.T
        else:
            logits = self.head(features)

        if return_features:
            return logits, features
        return logits

    def zero_shot_predict(
        self, x: torch.Tensor, class_names: List[str]
    ) -> torch.Tensor:
        """
        Zero-shot prediction with custom class names.

        Useful for experimenting with different prompts.
        """
        return self.encoder.zero_shot_classify(x, class_names)


# ============================================
# Model Registry
# ============================================

# Quick access to common model configurations
VLM_MODELS = {
    # OpenCLIP models
    "openclip_vit_b_32": ("openclip", "ViT-B-32", {"pretrained": "openai"}),
    "openclip_vit_b_16": ("openclip", "ViT-B-16", {"pretrained": "openai"}),
    "openclip_vit_l_14": ("openclip", "ViT-L-14", {"pretrained": "openai"}),
    "openclip_vit_l_14_336": ("openclip", "ViT-L-14-336", {"pretrained": "openai"}),
    "openclip_vit_h_14_laion": (
        "openclip",
        "ViT-H-14",
        {"pretrained": "laion2b_s32b_b79k"},
    ),
    "openclip_vit_g_14_laion": (
        "openclip",
        "ViT-G-14",
        {"pretrained": "laion2b_s34b_b88k"},
    ),
    # SigLIP models
    "siglip_base_224": ("siglip", "siglip-base-patch16-224", {}),
    "siglip_base_384": ("siglip", "siglip-base-patch16-384", {}),
    "siglip_large_384": ("siglip", "siglip-large-patch16-384", {}),
    "siglip_so400m_384": ("siglip", "siglip-so400m-patch14-384", {}),
    # Perception Encoder models
    "pe_core_b16_224": (
        "perception_encoder",
        "PE-Core-B16-224",
        {"backend": "open_clip"},
    ),
    "pe_core_l14_336": (
        "perception_encoder",
        "PE-Core-L14-336",
        {"backend": "open_clip"},
    ),
    "pe_core_g14_448": (
        "perception_encoder",
        "PE-Core-G14-448",
        {"backend": "open_clip"},
    ),
}


def get_vlm_model(
    model_name: str,
    num_classes: int,
    head_type: str = "linear",
    device: str = "cuda",
    class_names: Optional[List[str]] = None,
    **kwargs,
) -> VLMEncoderWithHead:
    """
    Get a VLM model with classification head.

    Args:
        model_name: Model name from VLM_MODELS registry or custom
        num_classes: Number of output classes
        head_type: "linear", "mlp", or "zero_shot"
        device: Device to load on
        class_names: Required for zero_shot head
        **kwargs: Additional arguments

    Returns:
        VLMEncoderWithHead instance
    """
    if model_name in VLM_MODELS:
        encoder_type, arch, default_kwargs = VLM_MODELS[model_name]
        final_kwargs = {**default_kwargs, **kwargs}
        encoder = create_vlm_encoder(encoder_type, arch, device=device, **final_kwargs)
    else:
        # Try to parse custom format: "encoder_type:arch"
        if ":" in model_name:
            encoder_type, arch = model_name.split(":", 1)
            encoder = create_vlm_encoder(encoder_type, arch, device=device, **kwargs)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. " f"Available: {list(VLM_MODELS.keys())}"
            )

    return VLMEncoderWithHead(
        encoder=encoder,
        num_classes=num_classes,
        head_type=head_type,
        class_names=class_names,
        **kwargs,
    )
