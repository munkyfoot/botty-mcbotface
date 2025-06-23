import io
import boto3, urllib.parse, os


class S3:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION")

    def public_upload(self, file_obj: io.BytesIO, key: str) -> str:
        self.s3.upload_fileobj(file_obj, self.bucket_name, key)
        return self.get_public_url(key)

    def get_public_url(self, key: str) -> str:
        safe_key = urllib.parse.quote_plus(key)
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{safe_key}"
