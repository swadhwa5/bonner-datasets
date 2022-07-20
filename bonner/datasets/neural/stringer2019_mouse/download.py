from pathlib import Path

from ...utils.figshare import get_url_dict
from ...utils import download_file

FIGSHARE_ARTICLE_ID = 6845348


def download_dataset(force_download: bool = False, **kwargs) -> None:
    urls = get_url_dict(FIGSHARE_ARTICLE_ID)
    urls_data = {key: url for key, url in urls.items() if "natimg2800_M" in key}
    for filename, url in urls_data.items():
        download_file(url, filepath=Path(filename), force=force_download)
    download_file(
        urls["images_natimg2800_all.mat"], filepath=Path(filename), force=force_download
    )
