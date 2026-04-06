"""
Tag Extraction Mitigator for vb-mitigator.

This mitigator extracts comprehensive tags from dataset images and classifies
them as relevant or irrelevant based on the task description.

Usage:
    python main.py --config configs/tag_extraction/tag_extraction_waterbirds.yaml

Config:
    MITIGATOR:
      TYPE: "tag_extraction"
      TAG_EXTRACTION:
        TASK_DESCRIPTION: "bird species classification"
        VLM_MODEL: "llava"
        LLM_MODEL: "ollama"
        ...
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
from tqdm import tqdm
from PIL import Image

from .base_trainer import BaseTrainer

# Import tag extraction pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from tag_extraction import (
    TagExtractionConfig,
    TagExtractionPipeline,
)


class TagExtractionTrainer(BaseTrainer):
    """
    Tag Extraction as a vb-mitigator mitigator.

    This is a preprocessing mitigator that:
    1. Extracts comprehensive tags from all images using a VLM
    2. Classifies tags as relevant/irrelevant using an LLM
    3. Outputs train_tags.csv for use with SAE mitigators

    It does NOT train a model - it prepares the tag annotations.
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """No model setup needed - we use external VLM/LLM."""
        self.model = None

    def _setup_optimizer(self):
        """SAE has its own optimizer."""
        self.optimizer = None

    def _setup_scheduler(self):
        """SAE has its own scheduler."""
        self.scheduler = None

    def _method_specific_setups(self):
        """Setup tag extraction pipeline."""
        tag_cfg = self.cfg.MITIGATOR.TAG_EXTRACTION

        # Determine output directory
        output_dir = tag_cfg.OUTPUT_DIR
        if not output_dir:
            output_dir = os.path.join(self.data_root, "tag_extraction_output")

        # Create config
        self.tag_config = TagExtractionConfig(
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
            output_dir=output_dir,
        )

        self.pipeline = TagExtractionPipeline(self.tag_config)
        self.resume_stage = tag_cfg.get("RESUME_FROM_STAGE", 1)

    def _collect_image_info(self) -> tuple:
        """Collect image paths, indices, and targets from dataloader."""
        print("\nCollecting image information from dataset...")

        image_paths = []
        image_indices = []
        image_targets = []

        dataloader = self.dataloaders["test"]
        dataset = dataloader.dataset

        # Try different dataset structures
        if hasattr(dataset, "samples"):
            # ImageFolder style
            for idx, (path, target) in enumerate(dataset.samples):
                image_paths.append(path)
                image_indices.append(idx)
                image_targets.append(target)

        elif hasattr(dataset, "img_fpath_list"):
            # img_fpath = self.img_fpath_list[index]
            # label = self.obj_label[index]
            # bg_label = self.bg_label[index]
            # cooc_obj_label = self.co_occur_obj_label[index]
            for idx, (path, target) in enumerate(
                zip(dataset.img_fpath_list, dataset.obj_label)
            ):
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
            # DataFrame-based dataset (common in bias datasets)
            df = dataset.df
            data_dir = getattr(dataset, "data_dir", self.data_root)

            # Find image path column
            path_cols = ["img_filename", "path", "image_path", "filename", "img_path"]
            path_col = None
            for col in path_cols:
                if col in df.columns:
                    path_col = col
                    break

            if path_col is None:
                raise ValueError(
                    f"Could not find image path column. Available: {df.columns.tolist()}"
                )

            # Find target column
            target_cols = ["target", "label", "y", "class"]
            target_col = None
            for col in target_cols:
                if col in df.columns:
                    target_col = col
                    break

            for idx, row in df.iterrows():
                path = row[path_col]
                if not os.path.isabs(path):
                    path = os.path.join(data_dir, path)

                image_paths.append(path)
                image_indices.append(idx)

                if target_col:
                    image_targets.append(int(row[target_col]))
                else:
                    image_targets.append(0)

        elif hasattr(dataset, "data") and hasattr(dataset, "targets"):
            # CIFAR-style dataset
            # Need to save images temporarily
            import tempfile

            temp_dir = tempfile.mkdtemp()
            print(f"  Saving images to temp directory: {temp_dir}")

            for idx in tqdm(range(len(dataset)), desc="  Preparing images"):
                img_data = dataset.data[idx]
                target = dataset.targets[idx]

                # Convert to PIL and save
                if isinstance(img_data, torch.Tensor):
                    img_data = img_data.numpy()

                img = Image.fromarray(img_data)
                path = os.path.join(temp_dir, f"{idx:06d}.png")
                img.save(path)

                image_paths.append(path)
                image_indices.append(idx)
                image_targets.append(int(target))

        else:
            # Generic fallback - iterate through dataset
            print("  Using generic dataset iteration (may be slow)...")

            for idx in tqdm(range(len(dataset)), desc="  Collecting"):
                item = dataset[idx]

                # Try to get path
                if isinstance(item, dict):
                    if "path" in item:
                        path = item["path"]
                    elif "image_path" in item:
                        path = item["image_path"]
                    else:
                        raise ValueError("Cannot find image path in dataset item")

                    index = item.get("index", idx)
                    target = item.get("targets", item.get("target", 0))
                else:
                    raise ValueError("Unsupported dataset format")

                image_paths.append(path)
                image_indices.append(index)
                image_targets.append(int(target))

        print(f"  Collected {len(image_paths)} images")

        return image_paths, image_indices, image_targets

    def train(self):
        """Run tag extraction pipeline."""
        print(f"\n{'='*60}")
        print("Tag Extraction Mitigator")
        print(f"{'='*60}")
        print(f"Task: {self.tag_config.task_description}")
        print(f"VLM: {self.tag_config.vlm_model}")
        print(f"LLM: {self.tag_config.llm_model}")
        print(f"Output: {self.tag_config.output_dir}")

        # Collect image info
        image_paths, image_indices, image_targets = self._collect_image_info()

        # Run pipeline
        self.pipeline.run(
            image_paths=image_paths,
            image_indices=image_indices,
            image_targets=image_targets,
            task_description=self.tag_config.task_description,
            resume_from_stage=self.resume_stage,
        )

        # Copy output to data root
        output_csv = self.pipeline.final_dir / "train_tags.csv"
        target_csv = os.path.join(self.data_root, "train_tags.csv")

        import shutil

        shutil.copy(output_csv, target_csv)

        print(f"\n{'='*60}")
        print("Tag Extraction Complete!")
        print(f"{'='*60}")
        print(f"\nOutput files:")
        print(f"  Main CSV: {target_csv}")
        print(f"  Full output: {self.tag_config.output_dir}")
        print(f"\nNext steps:")
        print(
            f"  1. Review tags (optional): edit {self.pipeline.stage3_dir / 'tag_relevance_review.json'}"
        )
        print(
            f"  2. Train SAE: use tag_sae, tag_only_sae, or tag_only_sae_v2 mitigator"
        )

    def eval(self):
        """No evaluation for tag extraction."""
        print("Tag extraction is a preprocessing step - no evaluation needed.")
        print("Use the generated train_tags.csv with SAE mitigators.")


# ============================================
# Config defaults to add to cfg.py
# ============================================

CONFIG_DEFAULTS = """
# Add to configs/cfg.py:

from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------
# TAG EXTRACTION
# ---------------------------------------------------------------------
CFG.MITIGATOR.TAG_EXTRACTION = CN()

# Task description (REQUIRED)
CFG.MITIGATOR.TAG_EXTRACTION.TASK_DESCRIPTION = ""

# VLM settings
CFG.MITIGATOR.TAG_EXTRACTION.VLM_MODEL = "llava"  # llava, qwen-vl, internvl
CFG.MITIGATOR.TAG_EXTRACTION.VLM_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
CFG.MITIGATOR.TAG_EXTRACTION.VLM_DEVICE = "cuda"
CFG.MITIGATOR.TAG_EXTRACTION.VLM_BATCH_SIZE = 1

# LLM settings
CFG.MITIGATOR.TAG_EXTRACTION.LLM_MODEL = "ollama"  # gpt-4, ollama, llama3, local
CFG.MITIGATOR.TAG_EXTRACTION.LLM_MODEL_PATH = ""
CFG.MITIGATOR.TAG_EXTRACTION.LLM_API_KEY = ""
CFG.MITIGATOR.TAG_EXTRACTION.LLM_BASE_URL = ""
CFG.MITIGATOR.TAG_EXTRACTION.LLM_TAG_BATCH_SIZE = 100

# Processing
CFG.MITIGATOR.TAG_EXTRACTION.MIN_TAG_FREQUENCY = 5
CFG.MITIGATOR.TAG_EXTRACTION.ENABLE_HUMAN_REVIEW = False
CFG.MITIGATOR.TAG_EXTRACTION.OUTPUT_DIR = ""
CFG.MITIGATOR.TAG_EXTRACTION.RESUME_FROM_STAGE = 1
"""
