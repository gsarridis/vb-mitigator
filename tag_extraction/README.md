# Comprehensive Tag Extraction Pipeline

A generic pipeline for extracting **ALL visual semantics** from images as tags, then classifying them as **relevant** or **irrelevant** based on the classification task.

## Why This Pipeline?

Machine learning models often learn **spurious correlations** (biases) from training data. For example, a bird classifier might learn that "water background" = "waterbird" instead of learning actual bird features.

This pipeline helps identify and separate:
- **Relevant tags**: Intrinsic to the subject being classified (e.g., "feathers", "beak", "wings" for bird classification)
- **Irrelevant tags**: Context/environment that could cause bias (e.g., "water", "forest", "sunny")

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: EXTRACTION (Per Image)                                            │
│   Image → VLM (LLaVA/Qwen-VL) → 6 targeted queries → Comprehensive Tags    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: AGGREGATION (Dataset-Level)                                       │
│   All image tags → Unique vocabulary with frequencies                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: RELEVANCE CLASSIFICATION (Once, LLM-based)                        │
│   Vocabulary + Task Description → LLM → Relevant/Irrelevant split          │
│   [Optional: Human Review]                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: APPLY MAPPING (Per Image)                                         │
│   Image tags → Lookup → Output CSV with relevant_tags, irrelevant_tags     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Core dependencies
pip install torch torchvision transformers pillow tqdm

# For LLaVA
pip install transformers accelerate

# For Qwen-VL
pip install transformers accelerate

# For OpenAI LLM
pip install openai

# For Ollama (local LLM)
pip install ollama
# Also install Ollama: https://ollama.ai
```

## Quick Start

### Command Line

```bash
# Basic usage
python extract_tags.py \
    --image_dir ./data/cars/images \
    --task "car type classification" \
    --output_dir ./output

# With specific models
python extract_tags.py \
    --image_dir ./data/birds/images \
    --task "bird species classification" \
    --vlm llava \
    --vlm_path llava-hf/llava-1.5-13b-hf \
    --llm gpt-4 \
    --api_key $OPENAI_API_KEY \
    --output_dir ./output

# With human review
python extract_tags.py \
    --image_dir ./data/faces/images \
    --task "facial expression recognition" \
    --human_review \
    --output_dir ./output

# Resume from stage 3 (re-classify relevance)
python extract_tags.py \
    --image_dir ./data/cars/images \
    --task "car type classification" \
    --resume_stage 3 \
    --output_dir ./output
```

### Python API

```python
from tag_extraction import extract_tags_for_dataset

# Simple usage
output_csv = extract_tags_for_dataset(
    image_paths=["img1.jpg", "img2.jpg", ...],
    image_indices=[0, 1, ...],
    image_targets=[0, 1, ...],
    task_description="car type classification",
    vlm_model="llava",
    llm_model="ollama"
)

# With full config
from tag_extraction import TagExtractionConfig, TagExtractionPipeline

config = TagExtractionConfig(
    vlm_model="llava",
    vlm_model_path="llava-hf/llava-1.5-13b-hf",
    llm_model="gpt-4",
    llm_api_key="sk-...",
    min_tag_frequency=10,
    enable_human_review=True,
    output_dir="./output"
)

pipeline = TagExtractionPipeline(config)
pipeline.run(
    image_paths=paths,
    image_indices=indices,
    image_targets=targets,
    task_description="bird species classification"
)
```

## Task Descriptions

The **task description** is crucial - it tells the LLM what the classification task is, so it can determine what's relevant:

| Task | Description | Relevant Examples | Irrelevant Examples |
|------|-------------|-------------------|---------------------|
| Car type classification | Distinguishing between types of cars | sedan, wheel, metallic, 4 doors | parking lot, tree, sunny, road |
| Bird species classification | Distinguishing between bird species | bird, feathers, beak, wings, flying | water, branch, forest, sky |
| Facial expression recognition | Identifying emotions from faces | smile, frown, eyes, teeth | office, shirt, indoor, bright |
| Dog breed classification | Identifying dog breeds | dog, fur, ears, tail, paws | grass, park, leash, person |

## Output Files

```
output/
├── stage1_extraction/
│   ├── image_tags.json           # Raw tags per image
│   └── extraction_errors.json    # Any failed images
│
├── stage2_vocabulary/
│   ├── tag_vocabulary.json       # {tag: frequency}
│   └── tag_statistics.json       # Summary statistics
│
├── stage3_relevance/
│   ├── tag_relevance_auto.json   # LLM classification
│   ├── tag_relevance_review.json # For human review (optional)
│   └── tag_relevance_final.json  # Final mapping
│
└── final/
    └── train_tags.csv            # Ready for SAE training
```

### Output CSV Format

```csv
index,target,tags,relevant_tags,irrelevant_tags
0,0,car | red | sedan | parking lot | sunny,car | red | sedan,parking lot | sunny
1,1,suv | blue | large | highway | cloudy,suv | blue | large,highway | cloudy
```

## Human Review

When `--human_review` is enabled:

1. Pipeline creates `tag_relevance_review.json`
2. Edit this file to correct any misclassifications
3. Save and re-run with `--resume_stage 4`

Review file format:
```json
{
    "instructions": "Review and correct classifications...",
    "task": "car type classification",
    "relevant": {
        "sedan": "Describes a car type",
        "wheel": "Intrinsic car part",
        "red": "Color of the car"
    },
    "irrelevant": {
        "parking lot": "Background location",
        "sunny": "Weather condition",
        "tree": "Background object"
    }
}
```

## Supported Models

### VLM (Image → Tags)

| Model | ID | Notes |
|-------|-----|-------|
| LLaVA | `llava` | Good balance of speed/quality |
| Qwen-VL | `qwen-vl` | Strong multilingual support |
| InternVL | `internvl` | State-of-the-art quality |

### LLM (Relevance Classification)

| Model | ID | Notes |
|-------|-----|-------|
| GPT-4 | `gpt-4` | Best quality (requires API key) |
| GPT-3.5 | `gpt-3.5-turbo` | Good quality, faster |
| Ollama | `ollama` | Local, free |
| Llama 3 | `llama3` | Via Ollama |
| Local | `local` | Any HuggingFace model |

## Integration with vb-mitigator

Add to your config:

```yaml
MITIGATOR:
  TAG_EXTRACTION:
    TASK_DESCRIPTION: "car type classification"
    VLM_MODEL: "llava"
    VLM_MODEL_PATH: "llava-hf/llava-1.5-7b-hf"
    LLM_MODEL: "ollama"
    MIN_TAG_FREQUENCY: 5
    ENABLE_HUMAN_REVIEW: False
```

Then use in your trainer:

```python
from tag_extraction import TagExtractor

extractor = TagExtractor(cfg)
tags_csv = extractor.extract_from_dataloader(train_loader)
```

## VLM Queries

The pipeline asks 6 targeted questions to ensure comprehensive coverage:

1. **Objects & Parts**: "List ALL objects and their parts..."
2. **Attributes**: "List ALL visual attributes: colors, sizes, shapes..."
3. **Scene & Environment**: "Describe the scene, location, setting..."
4. **Actions & Relations**: "What actions, poses, or relationships..."
5. **Photo & Style**: "Describe lighting, weather, camera angle..."
6. **Additional**: "What else is visible that wasn't mentioned?"

## Tips

1. **Use a larger VLM** for better tag quality (e.g., llava-1.5-13b vs 7b)
2. **Use GPT-4** for best relevance classification
3. **Enable human review** for critical applications
4. **Set appropriate min_frequency** based on dataset size
5. **Task description matters** - be specific about what you're classifying

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{tag_extraction_pipeline,
    title={Comprehensive Tag Extraction for Visual Debiasing},
    year={2024},
    url={https://github.com/your-repo}
}
```