#!/usr/bin/env python
"""
Standalone Tag Extraction Script.

Extract comprehensive tags from a dataset and classify them as relevant/irrelevant.

Usage:
    # Basic usage with image directory
    python extract_tags.py \
        --image_dir ./data/cars/images \
        --task "car type classification" \
        --output_dir ./output
    
    # With custom models
    python extract_tags.py \
        --image_dir ./data/birds/images \
        --task "bird species classification" \
        --vlm llava \
        --vlm_path llava-hf/llava-1.5-13b-hf \
        --llm gpt-4 \
        --output_dir ./output
    
    # Resume from specific stage
    python extract_tags.py \
        --image_dir ./data/cars/images \
        --task "car type classification" \
        --output_dir ./output \
        --resume_stage 3  # Skip extraction, re-run classification
    
    # With human review
    python extract_tags.py \
        --image_dir ./data/faces/images \
        --task "facial expression recognition" \
        --human_review \
        --output_dir ./output
    
    # From CSV file with image paths
    python extract_tags.py \
        --csv ./data/metadata.csv \
        --image_col img_path \
        --index_col index \
        --target_col label \
        --task "car type classification" \
        --output_dir ./output
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


def collect_images_from_dir(image_dir: str) -> tuple:
    """Collect all images from a directory."""
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(f"**/*{ext}"))
        image_paths.extend(image_dir.glob(f"**/*{ext.upper()}"))

    image_paths = sorted([str(p) for p in image_paths])
    image_indices = list(range(len(image_paths)))
    image_targets = [0] * len(image_paths)

    return image_paths, image_indices, image_targets


def collect_images_from_csv(
    csv_path: str,
    image_col: str,
    index_col: Optional[str] = None,
    target_col: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> tuple:
    """Collect images from a CSV file."""
    df = pd.read_csv(csv_path)

    if base_dir:
        image_paths = [os.path.join(base_dir, p) for p in df[image_col]]
    else:
        image_paths = df[image_col].tolist()

    if index_col and index_col in df.columns:
        image_indices = df[index_col].tolist()
    else:
        image_indices = list(range(len(image_paths)))

    if target_col and target_col in df.columns:
        image_targets = df[target_col].tolist()
    else:
        image_targets = [0] * len(image_paths)

    return image_paths, image_indices, image_targets


def main():
    parser = argparse.ArgumentParser(
        description="Extract comprehensive tags from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source (either directory or CSV)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image_dir", type=str, help="Directory containing images"
    )
    input_group.add_argument("--csv", type=str, help="CSV file with image paths")

    # CSV options
    parser.add_argument(
        "--image_col",
        type=str,
        default="image_path",
        help="Column name for image paths in CSV",
    )
    parser.add_argument(
        "--index_col",
        type=str,
        default=None,
        help="Column name for image indices in CSV",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Column name for target labels in CSV",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory to prepend to image paths from CSV",
    )

    # Task description (required)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description (e.g., 'car type classification')",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tag_extraction_output",
        help="Output directory",
    )

    # VLM settings
    parser.add_argument(
        "--vlm",
        type=str,
        default="llava",
        choices=["llava", "qwen-vl", "internvl"],
        help="VLM model for tag extraction",
    )
    parser.add_argument(
        "--vlm_path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="VLM model path or HuggingFace ID",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for VLM (cuda or cpu)"
    )

    # LLM settings
    parser.add_argument(
        "--llm",
        type=str,
        default="ollama",
        choices=[
            "gpt-4",
            "gpt-4o",
            "gpt-3.5-turbo",
            "ollama",
            "llama3",
            "mistral",
            "local",
        ],
        help="LLM for relevance classification",
    )
    parser.add_argument(
        "--llm_path", type=str, default="", help="LLM model path (for local models)"
    )
    parser.add_argument(
        "--api_key", type=str, default="", help="API key for OpenAI models"
    )

    # Processing options
    parser.add_argument(
        "--min_freq", type=int, default=5, help="Minimum tag frequency threshold"
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Number of tags per LLM call"
    )

    # Workflow options
    parser.add_argument(
        "--human_review", action="store_true", help="Enable human review step"
    )
    parser.add_argument(
        "--resume_stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Stage to resume from (1=extract, 2=aggregate, 3=classify, 4=apply)",
    )

    # Verbosity
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Collect images
    print("=" * 60)
    print("Comprehensive Tag Extraction Pipeline")
    print("=" * 60)

    if args.image_dir:
        print(f"\nCollecting images from: {args.image_dir}")
        image_paths, image_indices, image_targets = collect_images_from_dir(
            args.image_dir
        )
    else:
        print(f"\nReading images from CSV: {args.csv}")
        image_paths, image_indices, image_targets = collect_images_from_csv(
            args.csv, args.image_col, args.index_col, args.target_col, args.base_dir
        )

    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("Error: No images found!")
        sys.exit(1)

    # Import pipeline (do this after args parsing for faster --help)
    from tag_extraction_pipeline import TagExtractionConfig, TagExtractionPipeline

    # Create config
    config = TagExtractionConfig(
        vlm_model=args.vlm,
        vlm_model_path=args.vlm_path,
        vlm_device=args.device,
        llm_model=args.llm,
        llm_model_path=args.llm_path,
        llm_api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
        llm_tag_batch_size=args.batch_size,
        min_tag_frequency=args.min_freq,
        task_description=args.task,
        enable_human_review=args.human_review,
        output_dir=args.output_dir,
    )

    # Run pipeline
    pipeline = TagExtractionPipeline(config)

    try:
        pipeline.run(
            image_paths=image_paths,
            image_indices=image_indices,
            image_targets=image_targets,
            task_description=args.task,
            resume_from_stage=args.resume_stage,
        )

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"\nOutput files:")
        print(f"  Tags CSV:     {pipeline.final_dir / 'train_tags.csv'}")
        print(f"  Vocabulary:   {pipeline.stage2_dir / 'tag_vocabulary.json'}")
        print(f"  Relevance:    {pipeline.stage3_dir / 'tag_relevance_final.json'}")

        if args.human_review:
            print(
                f"\n  Review file:  {pipeline.stage3_dir / 'tag_relevance_review.json'}"
            )
            print(
                "  (Edit this file and re-run with --resume_stage 4 to apply changes)"
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted! You can resume with --resume_stage")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
