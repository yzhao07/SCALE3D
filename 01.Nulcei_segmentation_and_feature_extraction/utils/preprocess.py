import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from skimage.filters import sobel
import skimage as sk
from skimage import measure,img_as_float


def PreprocessNucleiChannel(img,median_kernel=2):
    """
    Preprocess image
    Args:
        img: numpy array Depth * Height * Width
    Returns:
        normalized_image: numpy array Depth * Height * Width
    """
    print("Preprocessing image...")
    print("Median kernel...",median_kernel)
    ## denoise
    normalized_image = sk.exposure.rescale_intensity(img, in_range='image', out_range=(0, 1))
    smoothed = gaussian_filter(normalized_image, sigma=1)
    laplacian = laplace(smoothed)
    normalized_image = normalized_image - laplacian
    kernel = sk.morphology.ball(median_kernel)
    normalized_image = sk.filters.median(normalized_image, kernel)
    normalized_image= sk.exposure.equalize_adapthist(normalized_image,clip_limit=0.01)
    
    return normalized_image
    