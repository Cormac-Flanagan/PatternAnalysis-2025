import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
import numpy as np
import nibabel as nib
from tqdm import tqdm, utils

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1
    return res


def load_data_3D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype: Union[np.dtype, type] = np.float32,
    getAffines=False,
    orient=False,
    early_stop=False,
):
    """
    Load medical image data from names , cases list provided into a list for each.
    This function pre - allocates 5 D arrays for conv3d to avoid excessive memory usage.
    normImage : bool ( normalise the image 0.0 -1.0)
    orient : Apply orientation and resample image ? Good for images with large slice &
    thickness or anisotropic resolution
    dtype : Type of the data . If dtype = np . uint8 , it is assumed that the data is &
    labels
    early_stop : Stop loading pre - maturely ? Leaves arrays mostly empty , for quick &
    loading and testing scripts .
    """
    affines = []

    interp = "linear"
    if dtype == np.uint8:
        interp = "nearest"

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])

    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]

    if categorical:
        first_case = to_channels(first_case)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]
        inImage = inImage[:, :, :depth]
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)

            images[
                i,
                : inImage.shape[0],
                : inImage.shape[1],
                : inImage.shape[2],
                : inImage.shape[3],
            ] = inImage
        else:
            images[
                i,
                : inImage.shape[0],
                : inImage.shape[1],
                : inImage.shape[2],
            ] = inImage

        affines.append(affine)
    if getAffines:
        return images, affines
    else:
        return images

def plot_results(input_path: str, output_path: str = "results.png"):
    """
    Reads a CSV file and plots its columns.

    Expected format:
        row_number, value1, [optional value2]

    Args:
        input_path (str): Path to the CSV file.
        output_path (str): Path to save the plot.
    """
    # Read CSV (no header, handle missing values)
    df = pd.read_csv(input_path, header=None)
    num_cols = df.shape[1]

    # Column 0 = x-axis (index)
    x = df.iloc[:, 0]

    plt.figure(figsize=(6, 4))

    # Plot first numeric column (index 1)
    plt.plot(x, df.iloc[:, 1], label="Training", marker="o")

    # Plot optional column if it exists
    if num_cols > 2:
        plt.plot(x, df.iloc[:, 2], label="Validation", marker="x")

    plt.xlabel("epoch")
    plt.ylabel("Average Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")
