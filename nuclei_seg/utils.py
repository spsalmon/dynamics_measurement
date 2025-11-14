from tifffile import imread
import numpy as np
from typing import Optional

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