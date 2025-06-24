import base64
import io
import re
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


def chunk_text(text: str, max_length: int) -> list[str]:
    """
    Break up a blob of text into multiple chunks, each with a maximum length of max_length.
    Uses a hierarchy of delimiters to break up chunks cleanly when possible: '\n', '. ', ', '.

    Parameters
    ----------
    text : str
        The input text to be chunked.
    max_length : int
        The maximum length of each chunk.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    delimiters = ["\n", ". ", ", "]

    def _split(text, max_length, delimiters):
        if len(text) <= max_length:
            return [text]
        if not delimiters:
            # No more delimiters, just split hard
            return [text[i : i + max_length] for i in range(0, len(text), max_length)]
        delim = delimiters[0]
        parts = []
        start = 0
        while start < len(text):
            # Find the furthest delimiter within max_length
            end = min(start + max_length, len(text))
            chunk = text[start:end]
            split_idx = chunk.rfind(delim)
            if split_idx == -1 or (start + split_idx + len(delim) - start) < 1:
                # No delimiter found, or it's at the start, try next delimiter
                if len(delimiters) > 1:
                    # Try with next delimiter
                    subchunks = _split(chunk, max_length, delimiters[1:])
                    parts.extend(subchunks)
                    start += len(chunk)
                else:
                    # No more delimiters, hard split
                    parts.append(chunk)
                    start += len(chunk)
            else:
                # Found a delimiter, split here
                split_point = start + split_idx + len(delim)
                parts.append(text[start:split_point])
                start = split_point
        return parts

    # Remove leading/trailing whitespace from each chunk
    return [
        chunk.strip() for chunk in _split(text, max_length, delimiters) if chunk.strip()
    ]
