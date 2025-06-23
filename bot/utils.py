import base64
import io
from PIL import Image


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
