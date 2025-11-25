"""Cloud storage abstraction for R2 and S3."""

import os
from abc import ABC, abstractmethod
from typing import Optional
from uuid import uuid4

import boto3


class StorageProvider(ABC):
    """Abstract base class for cloud storage providers."""

    @abstractmethod
    def public_upload(self, data: bytes, content_type: str = "image/jpeg") -> str:
        """Upload data and return a public URL.

        Args:
            data: The file data to upload.
            content_type: The MIME type of the file.

        Returns:
            A public URL to access the uploaded file.
        """
        pass

    @abstractmethod
    def get_public_url(self, key: str) -> str:
        """Get the public URL for a file.

        Args:
            key: The storage key (filename) of the file.

        Returns:
            A public URL to access the file.
        """
        pass


class R2Storage(StorageProvider):
    """Cloudflare R2 storage provider."""

    def __init__(self):
        """Initialize R2 client from environment variables.

        Required environment variables:
            R2_ACCESS_KEY_ID: Cloudflare R2 access key ID
            R2_SECRET_ACCESS_KEY: Cloudflare R2 secret access key
            R2_BUCKET_NAME: R2 bucket name
            R2_ACCOUNT_ID: Cloudflare account ID
            R2_PUBLIC_URL: Public URL base for the R2 bucket (e.g., custom domain or r2.dev URL)
        """
        required_vars = [
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET_NAME",
            "R2_ACCOUNT_ID",
            "R2_PUBLIC_URL",
        ]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required R2 environment variables: {', '.join(missing)}")

        self.bucket_name = os.getenv("R2_BUCKET_NAME")
        self.account_id = os.getenv("R2_ACCOUNT_ID")
        self.public_url = os.getenv("R2_PUBLIC_URL", "").rstrip("/")

        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{self.account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
        )

    def public_upload(self, data: bytes, content_type: str = "image/jpeg") -> str:
        """Upload data to R2 and return a public URL.

        Args:
            data: The file data to upload.
            content_type: The MIME type of the file.

        Returns:
            A public URL to access the uploaded file.
        """
        key = f"{uuid4()}.jpg"
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return self.get_public_url(key)

    def get_public_url(self, key: str) -> str:
        """Get the public URL for a file in R2.

        Args:
            key: The storage key (filename) of the file.

        Returns:
            A public URL to access the file.
        """
        return f"{self.public_url}/{key}"


class S3Storage(StorageProvider):
    """AWS S3 storage provider."""

    def __init__(self):
        """Initialize S3 client from environment variables.

        Required environment variables:
            AWS_ACCESS_KEY_ID: AWS access key ID
            AWS_SECRET_ACCESS_KEY: AWS secret access key
            S3_BUCKET_NAME: S3 bucket name
            AWS_REGION: AWS region (defaults to us-east-1)
        """
        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "S3_BUCKET_NAME",
        ]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required S3 environment variables: {', '.join(missing)}")

        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION", "us-east-1")

        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=self.region,
        )

    def public_upload(self, data: bytes, content_type: str = "image/jpeg") -> str:
        """Upload data to S3 and return a public URL.

        Args:
            data: The file data to upload.
            content_type: The MIME type of the file.

        Returns:
            A public URL to access the uploaded file.
        """
        key = f"{uuid4()}.jpg"
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return self.get_public_url(key)

    def get_public_url(self, key: str) -> str:
        """Get the public URL for a file in S3.

        Args:
            key: The storage key (filename) of the file.

        Returns:
            A public URL to access the file.
        """
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{key}"


def create_storage() -> StorageProvider:
    """Create a storage provider based on available environment variables.

    Checks for R2 configuration first (recommended), then falls back to S3.
    Raises an error if neither is configured.

    Returns:
        A configured StorageProvider instance.

    Raises:
        ValueError: If neither R2 nor S3 is properly configured.
    """
    # Try R2 first (recommended)
    r2_vars = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_ACCOUNT_ID", "R2_PUBLIC_URL"]
    if all(os.getenv(var) for var in r2_vars):
        return R2Storage()

    # Try S3 as fallback
    s3_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
    if all(os.getenv(var) for var in s3_vars):
        return S3Storage()

    # Neither configured - provide helpful error message
    raise ValueError(
        "Cloud storage is required but not configured. "
        "Please configure either Cloudflare R2 (recommended) or AWS S3.\n\n"
        "For R2, set: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_ACCOUNT_ID, R2_PUBLIC_URL\n"
        "For S3, set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME\n\n"
        "See README.md for detailed setup instructions."
    )
