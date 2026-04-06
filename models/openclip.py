"""
OpenCLIP Vision Encoder with Classification Head for VB-Mitigator.

This module provides a wrapper around OpenCLIP vision encoders that can be used
as a pretrained feature extractor with a trainable classification head for
downstream bias mitigation tasks.

Supported models include various ViT architectures from OpenCLIP:
- ViT-B/32, ViT-B/16, ViT-L/14, ViT-H/14, ViT-G/14
- Various pretrained weights (openai, laion2b, datacomp, etc.)

Usage:
    model = OpenCLIPEncoder(num_classes=2, model_name='ViT-B-32', pretrained='openai')
    logits, features = model(images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip

    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("Warning: open_clip not installed. Install with: pip install open_clip_torch")


# Available OpenCLIP models and their embedding dimensions
OPENCLIP_MODELS = {
    # Model name: (embed_dim, default_pretrained)
    "ViT-B-32": (512, "openai"),
    "ViT-B-16": (512, "openai"),
    "ViT-L-14": (768, "openai"),
    "ViT-L-14-336": (768, "openai"),
    "ViT-H-14": (1024, "laion2b_s32b_b79k"),
    "ViT-G-14": (1024, "laion2b_s34b_b88k"),
    "ViT-bigG-14": (1280, "laion2b_s39b_b160k"),
    "convnext_base": (512, "laion400m_s13b_b51k"),
    "convnext_base_w": (640, "laion2b_s13b_b82k"),
    "convnext_large_d": (768, "laion2b_s26b_b102k"),
    "RN50": (1024, "openai"),
    "RN101": (512, "openai"),
    "RN50x4": (640, "openai"),
    "RN50x16": (768, "openai"),
    "RN50x64": (1024, "openai"),
}


class OpenCLIPEncoder(nn.Module):
    """
    OpenCLIP Vision Encoder with a trainable classification head.

    The vision encoder is frozen by default and only the classification head is trained.
    This allows using powerful pretrained CLIP features for downstream tasks.

    Args:
        num_classes (int): Number of output classes
        model_name (str): OpenCLIP model name (e.g., 'ViT-B-32', 'ViT-L-14')
        pretrained (str): Pretrained weights to use (e.g., 'openai', 'laion2b_s34b_b79k')
        freeze_backbone (bool): Whether to freeze the vision encoder (default: True)
        use_projection (bool): Whether to use the CLIP projection layer (default: True)
                              If False, uses the raw vision transformer output

    Example:
        >>> model = OpenCLIPEncoder(num_classes=2, model_name='ViT-B-32', pretrained='openai')
        >>> images = torch.randn(4, 3, 224, 224)
        >>> logits, features = model(images)
        >>> print(logits.shape, features.shape)
        torch.Size([4, 2]) torch.Size([4, 512])
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        freeze_backbone: bool = True,
        use_projection: bool = True,
    ):
        super().__init__()

        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "open_clip is required for OpenCLIPEncoder. "
                "Install with: pip install open_clip_torch"
            )

        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.use_projection = use_projection
        self.num_classes = num_classes

        # Load OpenCLIP model
        print(
            f"Loading OpenCLIP model: {model_name} with pretrained weights: {pretrained}"
        )
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # Get embedding dimension
        if model_name in OPENCLIP_MODELS:
            self.embed_size = OPENCLIP_MODELS[model_name][0]
        else:
            # Try to infer from model
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = self.clip_model.encode_image(dummy_input)
                self.embed_size = dummy_output.shape[-1]

        print(f"OpenCLIP embed_size: {self.embed_size}")

        # Extract vision encoder
        self.vision_encoder = self.clip_model.visual

        # Freeze backbone if requested
        if freeze_backbone:
            print("Freezing OpenCLIP vision encoder backbone")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.fc = nn.Linear(self.embed_size, num_classes)

        # Store preprocessing info for dataset
        self._image_size = self._get_image_size()

        print(
            f"OpenCLIPEncoder initialized: {model_name}, pretrained={pretrained}, "
            f"embed_size={self.embed_size}, num_classes={num_classes}, "
            f"freeze_backbone={freeze_backbone}"
        )

    def _get_image_size(self):
        """Get the expected input image size for the model."""
        # Most CLIP models use 224, but some variants use different sizes
        if "336" in self.model_name:
            return 336
        elif "384" in self.model_name:
            return 384
        else:
            return 224

    @property
    def image_size(self):
        """Return expected input image size."""
        return self._image_size

    def encode_image(self, x):
        """
        Encode images using the CLIP vision encoder.

        Args:
            x: Input images tensor of shape (B, C, H, W)

        Returns:
            Image features of shape (B, embed_size)
        """
        if self.use_projection:
            # Use the full CLIP encoding pipeline including projection
            features = self.clip_model.encode_image(x)
        else:
            # Use raw vision transformer output (before projection)
            features = self.vision_encoder(x)

        # Normalize features (CLIP convention)
        features = F.normalize(features, dim=-1)

        return features

    def forward(self, x, norm=False):
        """
        Forward pass: encode images and classify.

        Args:
            x: Input images tensor of shape (B, C, H, W)
            norm: Whether to L2 normalize features (default: False, already normalized)

        Returns:
            logits: Classification logits of shape (B, num_classes)
            features: Image features of shape (B, embed_size)
        """
        # Encode images
        features = self.encode_image(x)

        if norm:
            features = F.normalize(features, dim=-1)

        # Classify
        logits = self.fc(features)

        return logits, features

    def badd_forward(self, x, f, m, norm=False):
        """
        BAdd forward pass for bias mitigation.

        Args:
            x: Input images
            f: List of bias features to add
            m: Multiplier for bias features
            norm: Whether to normalize features
        """
        features = self.encode_image(x)

        if norm:
            features = F.normalize(features, dim=-1)

        # Add bias features
        total_f = torch.sum(torch.stack(f), dim=0)
        features = features + total_f * m

        logits = self.fc(features)
        return logits

    def mavias_forward(self, x, f, norm=False):
        """
        MAVIAS forward pass.

        Args:
            x: Input images
            f: Projected features
            norm: Whether to normalize features
        """
        features = self.encode_image(x)

        if norm:
            features = F.normalize(features, dim=-1)
            f = F.normalize(f, dim=-1)

        logits = self.fc(features)
        logits2 = self.fc(f)

        return logits, logits2

    def get_transform(self):
        """Return the preprocessing transform for this model."""
        return self.preprocess


class OpenCLIPEncoderWithHead(nn.Module):
    """
    OpenCLIP with a more flexible classification head (MLP instead of linear).

    Useful when you want a more expressive classifier on top of CLIP features.
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        freeze_backbone: bool = True,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "open_clip is required. Install with: pip install open_clip_torch"
            )

        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes

        # Load OpenCLIP model
        print(f"Loading OpenCLIP model: {model_name}")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # Get embedding dimension
        if model_name in OPENCLIP_MODELS:
            self.embed_size = OPENCLIP_MODELS[model_name][0]
        else:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = self.clip_model.encode_image(dummy_input)
                self.embed_size = dummy_output.shape[-1]

        # Freeze backbone
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # MLP classification head
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

        print(
            f"OpenCLIPEncoderWithHead: {model_name}, embed_size={self.embed_size}, "
            f"head_hidden_dim={head_hidden_dim}"
        )

    def forward(self, x, norm=False):
        features = self.clip_model.encode_image(x)
        features = F.normalize(features, dim=-1)

        if norm:
            features = F.normalize(features, dim=-1)

        logits = self.fc(features)
        return logits, features

    def badd_forward(self, x, f, m, norm=False):
        features = self.clip_model.encode_image(x)
        features = F.normalize(features, dim=-1)

        total_f = torch.sum(torch.stack(f), dim=0)
        features = features + total_f * m

        logits = self.fc(features)
        return logits

    def mavias_forward(self, x, f, norm=False):
        features = self.clip_model.encode_image(x)
        features = F.normalize(features, dim=-1)

        if norm:
            f = F.normalize(f, dim=-1)

        logits = self.fc(features)
        logits2 = self.fc(f)

        return logits, logits2


def list_openclip_models():
    """List all available OpenCLIP models and their pretrained weights."""
    if not OPEN_CLIP_AVAILABLE:
        print("open_clip not installed")
        return []

    return open_clip.list_pretrained()


def get_openclip_model_info(model_name):
    """Get information about a specific OpenCLIP model."""
    if model_name in OPENCLIP_MODELS:
        embed_dim, default_pretrained = OPENCLIP_MODELS[model_name]
        return {
            "model_name": model_name,
            "embed_dim": embed_dim,
            "default_pretrained": default_pretrained,
        }
    return None


# Wrapper functions for compatibility with vb-mitigator's get_model interface
def openclip_vit_b_32(num_classes, pretrained=True):
    """OpenCLIP ViT-B/32 model."""
    return OpenCLIPEncoder(
        num_classes=num_classes,
        model_name="ViT-B-32",
        pretrained="openai" if pretrained else None,
        freeze_backbone=True,
    )


def openclip_vit_b_16(num_classes, pretrained=True):
    """OpenCLIP ViT-B/16 model."""
    return OpenCLIPEncoder(
        num_classes=num_classes,
        model_name="ViT-B-16",
        pretrained="openai" if pretrained else None,
        freeze_backbone=True,
    )


def openclip_vit_l_14(num_classes, pretrained=True):
    """OpenCLIP ViT-L/14 model."""
    return OpenCLIPEncoder(
        num_classes=num_classes,
        model_name="ViT-L-14",
        pretrained="openai" if pretrained else None,
        freeze_backbone=True,
    )


def openclip_vit_h_14(num_classes, pretrained=True):
    """OpenCLIP ViT-H/14 model (LAION pretrained)."""
    return OpenCLIPEncoder(
        num_classes=num_classes,
        model_name="ViT-H-14",
        pretrained="laion2b_s32b_b79k" if pretrained else None,
        freeze_backbone=True,
    )


def openclip_vit_g_14(num_classes, pretrained=True):
    """OpenCLIP ViT-G/14 model (LAION pretrained)."""
    return OpenCLIPEncoder(
        num_classes=num_classes,
        model_name="ViT-G-14",
        pretrained="laion2b_s34b_b88k" if pretrained else None,
        freeze_backbone=True,
    )


# Factory function for creating models with custom configurations
def create_openclip_encoder(
    num_classes: int,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    freeze_backbone: bool = True,
    use_mlp_head: bool = False,
    head_hidden_dim: int = 512,
    head_dropout: float = 0.1,
):
    """
    Factory function to create OpenCLIP encoder models.

    Args:
        num_classes: Number of output classes
        model_name: OpenCLIP model name
        pretrained: Pretrained weights name
        freeze_backbone: Whether to freeze the vision encoder
        use_mlp_head: Whether to use MLP classification head (vs linear)
        head_hidden_dim: Hidden dimension for MLP head
        head_dropout: Dropout rate for MLP head

    Returns:
        OpenCLIP encoder model
    """
    if use_mlp_head:
        return OpenCLIPEncoderWithHead(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
        )
    else:
        return OpenCLIPEncoder(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
