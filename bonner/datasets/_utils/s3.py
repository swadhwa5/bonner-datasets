from pathlib import Path

from loguru import logger
import boto3


def download(
    filepath_s3: Path,
    *,
    bucket: str,
    filepath_local: Path = None,
    use_cached: bool = True,
) -> None:
    """Download file(s) from S3.

    :param filepath_S3: path of file in S3
    :param bucket: S3 bucket name
    :param filepath_local: local path of file
    :param use_cached: use existing file or re-download, defaults to True
    """
    if filepath_local is None:
        filepath_local = filepath_s3
    s3 = boto3.client("s3")
    if (not use_cached) or (not filepath_local.exists()):
        logger.info(
            f"Downloading {filepath_s3} from S3 bucket {bucket} to {filepath_local}"
        )
        filepath_local.parent.mkdir(exist_ok=True, parents=True)
        with open(filepath_local, "wb") as f:
            s3.download_fileobj(bucket, str(filepath_s3), f)
    else:
        logger.info(
            "Using previously downloaded file at"
            f" {filepath_local} instead of downloading {filepath_s3} from S3 bucket"
            f" {bucket}"
        )
