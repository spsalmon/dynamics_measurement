from tifffile import imread
import numpy as np
from typing import Optional
import shutil
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args

def read_tiff_file(
    file_path: str,
    channels_to_keep: Optional[list[int]] = None,
) -> np.ndarray:
    """
    Read a TIFF file and optionally select specific channels from the image.

    Parameters:
            file_path (str): Path to the TIFF image file.
            channels_to_keep (list): List of channel indices to keep. If empty or None, all channels are kept.

    Returns:
            np.ndarray: The image data as a NumPy array. The number of dimensions may vary depending on the input and selected channels.

    Raises:
            ValueError: If the image file cannot be read.
    """
    try:
        image = imread(file_path)
    except Exception as e:
        raise ValueError(f"Error while reading image file {file_path} : {e}")

    # If no channels are specified, return the image as is.
    if image.ndim == 2 or not channels_to_keep:
        return image

    if image.ndim == 3:
        return image[channels_to_keep, ...].squeeze()  # type: ignore
    else:
        return image[:, channels_to_keep, ...].squeeze()  # type: ignore
    
def copy_without_permissions(src, dst):
    """Copy file without preserving permissions"""
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
    return dst
    
def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm)
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    x = x/np.max(x)
    return x, y