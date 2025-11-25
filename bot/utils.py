import io
from PIL import Image
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .storage import StorageProvider


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


def prepare_image(
    image_data: bytes,
    storage: "StorageProvider",
    *,
    max_size: int = 1024,
    quality: int = 95,
) -> tuple[str, bytes]:
    """Compress an image and upload it to cloud storage.

    The helper applies sensible compression settings and uploads the result
    to the configured storage provider (R2 or S3), returning the public URL.

    Parameters
    ----------
    image_data: bytes
        The raw image bytes.
    storage: StorageProvider
        A configured storage provider instance (R2Storage or S3Storage).
    max_size: int, optional
        Maximum image dimension (width/height) after compression.
    quality: int, optional
        JPEG quality level (1-100).

    Returns
    -------
    tuple[str, bytes]
        ``(image_url, compressed_image_bytes)``
    """
    compressed = compress_image(image_data, max_size=max_size, quality=quality)
    image_url = storage.public_upload(compressed)
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
