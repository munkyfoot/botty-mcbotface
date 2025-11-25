"""Image model configurations for Replicate."""

from dataclasses import dataclass, field
from typing import Any, Literal

# Type aliases for aspect ratios
AspectRatio = Literal[
    "1:1",
    "4:3",
    "3:4",
    "16:9",
    "9:16",
    "3:2",
    "2:3",
    "21:9",
]


@dataclass
class ImageModelConfig:
    """Configuration for a Replicate image model."""

    # Model identifier on Replicate
    model_id: str

    # Human-readable name
    name: str

    # Default parameters for the model
    default_params: dict[str, Any] = field(default_factory=dict)

    # Supported aspect ratios (None means all standard ratios supported)
    supported_aspect_ratios: list[str] | None = None

    # Parameter mappings (maps our standard params to model-specific params)
    # e.g., {"size": "resolution"} means our "size" maps to model's "resolution"
    param_mappings: dict[str, str] = field(default_factory=dict)

    # Whether the model supports image editing
    supports_editing: bool = True

    # The key used for input images (varies by model)
    image_input_key: str = "image_input"

    # Whether input images should be passed as a list
    image_input_as_list: bool = True

    # Maximum number of input images supported
    max_input_images: int = 1


# Available image model configurations
IMAGE_MODELS: dict[str, ImageModelConfig] = {
    "seedream": ImageModelConfig(
        model_id="bytedance/seedream-4",
        name="Seedream 4",
        default_params={
            "size": "1K",
            "enhance_prompt": True,
        },
        image_input_key="image_input",
        image_input_as_list=True,
        max_input_images=10,  # Supports batch and multi-reference
    ),
    "nano-banana": ImageModelConfig(
        model_id="google/nano-banana",
        name="Nano Banana",
        default_params={
            "output_format": "png",
        },
        image_input_key="image_input",
        image_input_as_list=True,
        max_input_images=3,  # Supports up to 3 images for multi-image fusion
    ),
    "nano-banana-pro": ImageModelConfig(
        model_id="google/nano-banana-pro",
        name="Nano Banana Pro",
        default_params={
            "resolution": "2K",
            "output_format": "png",
            "safety_filter_level": "block_only_high",
        },
        image_input_key="image_input",
        image_input_as_list=True,
        max_input_images=14,  # Supports up to 14 images
    ),
}

# Default model key
DEFAULT_IMAGE_MODEL = "seedream"

# Currently active model (can be changed at runtime)
_active_model_key: str = DEFAULT_IMAGE_MODEL


def initialize_from_settings(model_key: str | None) -> None:
    """Initialize the image model from settings.

    Args:
        model_key: The model key from settings, or None to use default.

    Logs a warning if the key is invalid and falls back to default.
    """
    import logging

    global _active_model_key

    if model_key is None:
        _active_model_key = DEFAULT_IMAGE_MODEL
        return

    if model_key not in IMAGE_MODELS:
        valid_keys = ", ".join(IMAGE_MODELS.keys())
        logging.warning(
            "Invalid image_model '%s' in settings. Valid options: %s. Using default '%s'.",
            model_key,
            valid_keys,
            DEFAULT_IMAGE_MODEL,
        )
        _active_model_key = DEFAULT_IMAGE_MODEL
        return

    _active_model_key = model_key
    logging.info("Image model set to '%s' (%s)", model_key, IMAGE_MODELS[model_key].name)


def get_active_model() -> ImageModelConfig:
    """Get the currently active image model configuration."""
    return IMAGE_MODELS[_active_model_key]


def get_active_model_key() -> str:
    """Get the key of the currently active image model."""
    return _active_model_key


def set_active_model(key: str) -> ImageModelConfig:
    """Set the active image model by key.

    Args:
        key: The model key (e.g., "seedream", "nano-banana", "nano-banana-pro")

    Returns:
        The newly active ImageModelConfig

    Raises:
        ValueError: If the key is not a valid model key
    """
    global _active_model_key
    if key not in IMAGE_MODELS:
        valid_keys = ", ".join(IMAGE_MODELS.keys())
        raise ValueError(f"Invalid model key '{key}'. Valid keys: {valid_keys}")
    _active_model_key = key
    return IMAGE_MODELS[key]


def get_model(key: str) -> ImageModelConfig:
    """Get a specific model configuration by key.

    Args:
        key: The model key

    Returns:
        The ImageModelConfig for the specified key

    Raises:
        ValueError: If the key is not a valid model key
    """
    if key not in IMAGE_MODELS:
        valid_keys = ", ".join(IMAGE_MODELS.keys())
        raise ValueError(f"Invalid model key '{key}'. Valid keys: {valid_keys}")
    return IMAGE_MODELS[key]


def list_models() -> dict[str, str]:
    """List all available models.

    Returns:
        A dictionary mapping model keys to human-readable names
    """
    return {key: config.name for key, config in IMAGE_MODELS.items()}


def build_generation_params(
    prompt: str,
    aspect_ratio: AspectRatio = "1:1",
    model_key: str | None = None,
    **extra_params: Any,
) -> tuple[str, dict[str, Any]]:
    """Build parameters for image generation.

    Args:
        prompt: The generation prompt
        aspect_ratio: Desired aspect ratio
        model_key: Optional model key (uses active model if not specified)
        **extra_params: Additional model-specific parameters

    Returns:
        A tuple of (model_id, params_dict)
    """
    config = IMAGE_MODELS[model_key] if model_key else get_active_model()

    params = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        **config.default_params,
        **extra_params,
    }

    return config.model_id, params


def build_editing_params(
    prompt: str,
    image_input: Any | list[Any],
    aspect_ratio: AspectRatio | Literal["match_input_image"] = "match_input_image",
    model_key: str | None = None,
    **extra_params: Any,
) -> tuple[str, dict[str, Any]]:
    """Build parameters for image editing.

    Args:
        prompt: The editing prompt
        image_input: The input image(s) - single item or list of (bytes, BytesIO, or URL)
        aspect_ratio: Desired aspect ratio or "match_input_image"
        model_key: Optional model key (uses active model if not specified)
        **extra_params: Additional model-specific parameters

    Returns:
        A tuple of (model_id, params_dict)
    """
    config = IMAGE_MODELS[model_key] if model_key else get_active_model()

    # Normalize to list
    if isinstance(image_input, list):
        images = image_input
    else:
        images = [image_input]

    # Enforce max input images limit
    if len(images) > config.max_input_images:
        images = images[: config.max_input_images]

    # If model expects a list, pass as list; otherwise pass single item
    if config.image_input_as_list:
        image_value = images
    else:
        image_value = images[0] if images else None

    params = {
        "prompt": prompt,
        config.image_input_key: image_value,
        "aspect_ratio": aspect_ratio,
        **config.default_params,
        **extra_params,
    }

    return config.model_id, params
