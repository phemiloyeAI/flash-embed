import os 
from git import List
import webdataset as wds

def read_webdataset(path: str):
    """Reads a WebDataset from the specified path.

    Args:
        path (str): The path to the WebDataset.
    Returns:
        wds.WebDataset: The loaded WebDataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    try:
        dataset = (
            wds.WebDataset(path)
            .decode("pil")
            .to_tuple("jpg", "txt")
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read WebDataset: {e}")

    return dataset

def read_text_file(file_path: str) -> str:
    """Reads a text file and returns its content.

    Args:
        file_path (str): The path to the text file.
    Returns:
        str: The content of the text file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The specified file does not exist: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read text file: {e}")

    return content


def iter_images_from_shards(shard_paths: List[str]):
    """Iterates over images from multiple WebDataset shards.

    Args:
        shard_paths (List[str]): List of paths to the WebDataset shards.
    Yields:
        Tuple: A tuple containing the image and its corresponding text.
    """
    for shard_path in shard_paths:
        dataset = read_webdataset(shard_path)
        for sample in dataset:
            yield sample

def iter_images_from_directory(directory_path: str):
    """Iterates over image files in a directory.

    Args:
        directory_path (str): The path to the directory containing image files.
    Yields:
        PIL.Image.Image: The loaded image.
    """
    from PIL import Image

    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"The specified path is not a directory: {directory_path}")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                image = Image.open(file_path).convert("RGB")
                yield image
            except Exception as e:
                print(f"Failed to read image file {file_path}: {e}")
