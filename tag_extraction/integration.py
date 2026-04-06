"""
Tag Extraction Integration for VB-Mitigator.

This module integrates the tag extraction pipeline with the vb-mitigator framework.
It can be used as a preprocessing step or as part of the training pipeline.

Usage:
    # As standalone preprocessing
    python -m tag_extraction.extract_tags --config config.yaml

    # Or programmatically
    from tag_extraction import TagExtractor

    extractor = TagExtractor(cfg)
    extractor.extract_and_save()
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from PIL import Image

from .tag_extraction_pipeline import (
    TagExtractionConfig,
    TagExtractionPipeline,
    extract_tags_for_dataset,
)


class TagExtractor:
    """
    Tag extractor integrated with vb-mitigator.

    Extracts comprehensive tags from a dataset and classifies them
    as relevant or irrelevant based on the task description.
    """

    def __init__(self, cfg):
        """
        Initialize with vb-mitigator config.

        Args:
            cfg: vb-mitigator configuration object
        """
        self.cfg = cfg
        self.data_root = cfg.DATASET.ROOT

        # Get tag extraction config
        tag_cfg = cfg.MITIGATOR.TAG_EXTRACTION

        self.config = TagExtractionConfig(
            vlm_model=tag_cfg.VLM_MODEL,
            vlm_model_path=tag_cfg.VLM_MODEL_PATH,
            vlm_device=tag_cfg.get("VLM_DEVICE", "cuda"),
            llm_model=tag_cfg.LLM_MODEL,
            llm_model_path=tag_cfg.get("LLM_MODEL_PATH", ""),
            llm_api_key=tag_cfg.get("LLM_API_KEY", ""),
            llm_base_url=tag_cfg.get("LLM_BASE_URL", ""),
            vlm_batch_size=tag_cfg.get("VLM_BATCH_SIZE", 1),
            llm_tag_batch_size=tag_cfg.get("LLM_TAG_BATCH_SIZE", 100),
            min_tag_frequency=tag_cfg.get("MIN_TAG_FREQUENCY", 5),
            task_description=tag_cfg.TASK_DESCRIPTION,
            enable_human_review=tag_cfg.get("ENABLE_HUMAN_REVIEW", False),
            output_dir=os.path.join(self.data_root, "tag_extraction_output"),
        )

        self.pipeline = TagExtractionPipeline(self.config)

    def extract_from_dataloader(
        self,
        dataloader,
        task_description: Optional[str] = None,
        resume_from_stage: int = 1,
    ) -> str:
        """
        Extract tags from a dataloader.

        Args:
            dataloader: PyTorch dataloader
            task_description: Override task description
            resume_from_stage: Stage to resume from

        Returns:
            Path to output CSV file
        """
        # Collect image paths, indices, and targets
        image_paths = []
        image_indices = []
        image_targets = []

        dataset = dataloader.dataset

        print("Collecting image information from dataset...")

        # Try different dataset structures
        if hasattr(dataset, "samples"):
            # ImageFolder style
            for idx, (path, target) in enumerate(dataset.samples):
                image_paths.append(path)
                image_indices.append(idx)
                image_targets.append(target)

        elif hasattr(dataset, "imgs"):
            # Another ImageFolder style
            for idx, (path, target) in enumerate(dataset.imgs):
                image_paths.append(path)
                image_indices.append(idx)
                image_targets.append(target)

        elif hasattr(dataset, "df"):
            # DataFrame-based dataset
            df = dataset.df
            data_dir = getattr(dataset, "data_dir", self.data_root)

            for idx, row in df.iterrows():
                if "img_filename" in df.columns:
                    path = os.path.join(data_dir, row["img_filename"])
                elif "path" in df.columns:
                    path = os.path.join(data_dir, row["path"])
                elif "image_path" in df.columns:
                    path = row["image_path"]
                else:
                    raise ValueError("Could not find image path column in dataframe")

                image_paths.append(path)
                image_indices.append(idx)

                if "target" in df.columns:
                    image_targets.append(row["target"])
                elif "label" in df.columns:
                    image_targets.append(row["label"])
                else:
                    image_targets.append(0)

        elif hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
            # Generic dataset - iterate through
            for idx in tqdm(range(len(dataset)), desc="Collecting dataset info"):
                item = dataset[idx]

                # Try to get image path
                if "path" in item:
                    path = item["path"]
                elif "image_path" in item:
                    path = item["image_path"]
                elif hasattr(dataset, "get_image_path"):
                    path = dataset.get_image_path(idx)
                else:
                    raise ValueError("Could not determine image path from dataset")

                image_paths.append(path)
                image_indices.append(item.get("index", idx))
                image_targets.append(item.get("targets", item.get("target", 0)))
        else:
            raise ValueError("Unsupported dataset structure")

        print(f"Collected {len(image_paths)} images")

        # Run pipeline
        task = task_description or self.config.task_description

        self.pipeline.run(
            image_paths=image_paths,
            image_indices=image_indices,
            image_targets=image_targets,
            task_description=task,
            resume_from_stage=resume_from_stage,
        )

        return str(self.pipeline.final_dir / "train_tags.csv")

    def extract_from_paths(
        self,
        image_paths: List[str],
        image_indices: Optional[List[int]] = None,
        image_targets: Optional[List[int]] = None,
        task_description: Optional[str] = None,
        resume_from_stage: int = 1,
    ) -> str:
        """
        Extract tags from a list of image paths.

        Args:
            image_paths: List of image file paths
            image_indices: Optional list of indices (defaults to 0, 1, 2, ...)
            image_targets: Optional list of target labels (defaults to 0)
            task_description: Override task description
            resume_from_stage: Stage to resume from

        Returns:
            Path to output CSV file
        """
        if image_indices is None:
            image_indices = list(range(len(image_paths)))

        if image_targets is None:
            image_targets = [0] * len(image_paths)

        task = task_description or self.config.task_description

        self.pipeline.run(
            image_paths=image_paths,
            image_indices=image_indices,
            image_targets=image_targets,
            task_description=task,
            resume_from_stage=resume_from_stage,
        )

        return str(self.pipeline.final_dir / "train_tags.csv")


# ============================================
# Config Defaults for cfg.py
# ============================================

CONFIG_DEFAULTS = """
# ============================================
# Add to configs/cfg.py
# ============================================

# ---------------------------------------------------------------------
# TAG EXTRACTION (Comprehensive Semantic Extraction)
# ---------------------------------------------------------------------
CFG.MITIGATOR.TAG_EXTRACTION = CN()

# ---- VLM Settings (for image → tags) ----
# Supported: "llava", "qwen-vl", "internvl"
CFG.MITIGATOR.TAG_EXTRACTION.VLM_MODEL = "llava"
CFG.MITIGATOR.TAG_EXTRACTION.VLM_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
CFG.MITIGATOR.TAG_EXTRACTION.VLM_DEVICE = "cuda"
CFG.MITIGATOR.TAG_EXTRACTION.VLM_BATCH_SIZE = 1

# ---- LLM Settings (for relevance classification) ----
# Supported: "gpt-4", "gpt-3.5-turbo", "ollama", "llama3", "local"
CFG.MITIGATOR.TAG_EXTRACTION.LLM_MODEL = "ollama"
CFG.MITIGATOR.TAG_EXTRACTION.LLM_MODEL_PATH = ""
CFG.MITIGATOR.TAG_EXTRACTION.LLM_API_KEY = ""  # For OpenAI
CFG.MITIGATOR.TAG_EXTRACTION.LLM_BASE_URL = ""  # For custom endpoints
CFG.MITIGATOR.TAG_EXTRACTION.LLM_TAG_BATCH_SIZE = 100

# ---- Task Description ----
# Describe the classification task (e.g., "car type classification")
# The LLM uses this to determine what's relevant vs irrelevant
CFG.MITIGATOR.TAG_EXTRACTION.TASK_DESCRIPTION = ""

# ---- Processing ----
CFG.MITIGATOR.TAG_EXTRACTION.MIN_TAG_FREQUENCY = 5

# ---- Human Review ----
CFG.MITIGATOR.TAG_EXTRACTION.ENABLE_HUMAN_REVIEW = False

# ---- Output ----
CFG.MITIGATOR.TAG_EXTRACTION.OUTPUT_DIR = ""  # Defaults to {data_root}/tag_extraction_output
"""


# ============================================
# CLI Entry Point
# ============================================


def main():
    """Command-line interface for tag extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract comprehensive tags from images"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description (e.g., 'car type classification')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tag_extraction_output",
        help="Output directory",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="llava",
        choices=["llava", "qwen-vl", "internvl"],
        help="VLM to use for tag extraction",
    )
    parser.add_argument(
        "--vlm_path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="VLM model path",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="ollama",
        choices=["gpt-4", "gpt-3.5-turbo", "ollama", "llama3", "local"],
        help="LLM to use for relevance classification",
    )
    parser.add_argument(
        "--llm_path", type=str, default="", help="LLM model path (for local models)"
    )
    parser.add_argument("--min_freq", type=int, default=5, help="Minimum tag frequency")
    parser.add_argument(
        "--human_review", action="store_true", help="Enable human review step"
    )
    parser.add_argument(
        "--resume_stage", type=int, default=1, help="Stage to resume from (1-4)"
    )

    args = parser.parse_args()

    # Collect image paths
    image_dir = Path(args.image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(f"**/*{ext}"))
        image_paths.extend(image_dir.glob(f"**/*{ext.upper()}"))

    image_paths = sorted([str(p) for p in image_paths])
    image_indices = list(range(len(image_paths)))
    image_targets = [0] * len(image_paths)  # Default targets

    print(f"Found {len(image_paths)} images")

    # Run pipeline
    output_csv = extract_tags_for_dataset(
        image_paths=image_paths,
        image_indices=image_indices,
        image_targets=image_targets,
        task_description=args.task,
        output_dir=args.output_dir,
        vlm_model=args.vlm,
        vlm_model_path=args.vlm_path,
        llm_model=args.llm,
        llm_model_path=args.llm_path,
        enable_human_review=args.human_review,
        min_tag_frequency=args.min_freq,
        resume_from_stage=args.resume_stage,
    )

    print(f"\nOutput CSV: {output_csv}")


if __name__ == "__main__":
    main()
