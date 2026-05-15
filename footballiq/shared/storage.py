import os
from typing import BinaryIO

import boto3
from botocore.client import Config


def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{os.environ['MINIO_ENDPOINT']}",
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_file(bucket: str, key: str, file_obj: BinaryIO, content_type: str = "video/mp4") -> str:
    client = get_minio_client()
    client.upload_fileobj(file_obj, bucket, key, ExtraArgs={"ContentType": content_type})
    return f"http://{os.environ['MINIO_ENDPOINT']}/{bucket}/{key}"


def download_file(bucket: str, key: str, dest_path: str) -> None:
    client = get_minio_client()
    client.download_file(bucket, key, dest_path)


def generate_presigned_url(bucket: str, key: str, expires: int = 3600) -> str:
    client = get_minio_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )
