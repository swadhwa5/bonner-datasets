from pathlib import Path

from PIL import Image
from scipy.io import loadmat

from .._figshare import get_url_dict
from .._utils import download_file

FIGSHARE_ARTICLE_ID = 6845348


def download_dataset(force_download: bool = False) -> None:
    urls = get_url_dict(FIGSHARE_ARTICLE_ID)
    urls_data = {key: url for key, url in urls.items() if "natimg2800_M" in key}
    for filename, url in urls_data.items():
        download_file(url, filepath=Path(filename), force=force_download)
    filename = "images_natimg2800_all.mat"
    download_file(urls[filename], filepath=Path(filename), force=force_download)


def save_images() -> list[Path]:
    images = loadmat("images_natimg2800_all.mat", simplify_cells=True)["imgs"]
    Path("images").mkdir(parents=True, exist_ok=True)
    paths = []
    for i_image in range(images.shape[-1]):
        path = Path("images") / f"image{i_image:04}.png"
        paths.append(path)
        Image.fromarray(images[:, :, i_image]).save(path)
    return paths
