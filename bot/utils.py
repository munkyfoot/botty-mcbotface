import base64
import io
from PIL import Image
from typing import TYPE_CHECKING, Tuple
import uuid

if TYPE_CHECKING:
    from .s3 import S3


def compress_image(image_data: bytes, max_size: int = 1024, quality: int = 95) -> bytes:
    """
    Compresses an image to a maximum size and quality.
    """
    image = Image.open(io.BytesIO(image_data))
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image = image.convert("RGB")  # Ensure JPEG compatible
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    compressed_data = buffer.getvalue()
    return compressed_data


def get_base64_image_url(image_data: bytes) -> str:
    """
    Encodes an image to base64.
    """
    return f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}"


def prepare_image(
    image_data: bytes,
    s3: "S3 | None" = None,
    key: str | None = None,
    *,
    s3_max_size: int = 1024,
    s3_quality: int = 95,
    base64_max_size: int = 512,
    base64_quality: int = 70,
) -> tuple[str, bytes]:
    """Compress an image and make it available via a URL.

    The helper applies sensible compression settings and then either uploads the
    result to S3 (returning the public URL) or returns a base-64 data-URL if no
    S3 client is provided.

    Parameters
    ----------
    image_data: bytes
        The raw image bytes.
    s3: S3 | None, optional
        An initialised :class:`~bot.s3.S3` instance. If *None*, a data-URL will
        be produced instead of an S3 URL.
    key: str | None, optional
        Destination key to use when uploading to S3. If omitted a UUID-based
        key will be generated.
    s3_max_size / s3_quality: int, optional
        Compression parameters used when uploading to S3.
    base64_max_size / base64_quality: int, optional
        Compression parameters used when falling back to a base-64 data URL.

    Returns
    -------
    tuple[str, bytes]
        ``(image_url, compressed_image_bytes)``
    """
    if s3:
        # Light compression for storage
        compressed = compress_image(
            image_data, max_size=s3_max_size, quality=s3_quality
        )
        if key is None:
            key = f"images/unsorted/{uuid.uuid4()}.jpg"
        image_url = s3.public_upload(io.BytesIO(compressed), key)
        return image_url, compressed

    # Fallback: data-URL (more aggressive compression)
    compressed = compress_image(
        image_data, max_size=base64_max_size, quality=base64_quality
    )
    image_url = get_base64_image_url(compressed)
    return image_url, compressed
