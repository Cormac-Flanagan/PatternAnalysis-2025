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
    dtype=np.float32,
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
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        inImage = niftiImage.get_fdata(caching="unchanged")
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]
        inImage = inImage[:, :, :depth]
        inImage = inImage.astype(dtype)

        if normImage:
            #
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
        if i > 20 and early_stop:
            break
    if getAffines:
        return images, affines
    else:
        return images


print(load_data_3D(["./data/semantic_MRs/B006_Week0_LFOV.nii.gz"], orient=True))
