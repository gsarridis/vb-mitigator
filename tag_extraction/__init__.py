"""
Comprehensive Tag Extraction Module.

This module provides a complete pipeline for:
1. Extracting ALL visual semantics from images as tags
2. Classifying tags as relevant or irrelevant based on the task

Usage:
    from tag_extraction import (
        TagExtractionPipeline,
        TagExtractionConfig,
        TagExtractor,
        extract_tags_for_dataset
    )

    # Simple usage
    output_csv = extract_tags_for_dataset(
        image_paths=paths,
        image_indices=indices,
        image_targets=targets,
        task_description="car type classification",
        vlm_model="llava",
        llm_model="ollama"
    )

    # Or with config
    config = TagExtractionConfig(
        vlm_model="llava",
        llm_model="gpt-4",
        task_description="bird species classification"
    )
    pipeline = TagExtractionPipeline(config)
    pipeline.run(...)
"""

from .tag_extraction_pipeline import (
    TagExtractionConfig,
    TagExtractionPipeline,
    extract_tags_for_dataset,
    BaseVLM,
    LLaVAVLM,
    QwenVLVLM,
    InternVLVLM,
    BaseLLM,
    OpenAILLM,
    OllamaLLM,
    TransformersLLM,
    create_vlm,
    create_llm,
    VLM_PROMPTS,
    LLM_RELEVANCE_PROMPT,
)

from .integration import (
    TagExtractor,
    CONFIG_DEFAULTS,
)

__all__ = [
    # Main classes
    "TagExtractionConfig",
    "TagExtractionPipeline",
    "TagExtractor",
    # Convenience function
    "extract_tags_for_dataset",
    # VLM classes
    "BaseVLM",
    "LLaVAVLM",
    "QwenVLVLM",
    "InternVLVLM",
    "create_vlm",
    # LLM classes
    "BaseLLM",
    "OpenAILLM",
    "OllamaLLM",
    "TransformersLLM",
    "create_llm",
    # Prompts (for customization)
    "VLM_PROMPTS",
    "LLM_RELEVANCE_PROMPT",
    # Config
    "CONFIG_DEFAULTS",
]
