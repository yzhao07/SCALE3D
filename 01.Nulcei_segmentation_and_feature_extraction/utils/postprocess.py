import skimage as sk
import numpy as np
from skimage.measure import regionprops_table

def postprocess(output_mask,min_size=150,max_size=7000):
    """
    Postprocess the output mask by removing small and large objects.

    Parameters: 
    output_mask (np.array): The input mask to be postprocessed.
    min_size (int): Minimum size of objects to keep.
    max_size (int): Maximum size of objects to keep.

    Returns:
    np.array: The postprocessed mask with objects within the specified size range.
    """
    mask = sk.segmentation.clear_border(output_mask.astype(int))
    properties = regionprops_table(mask, properties=('label', 'area'))
    # Determine which labels to keep
    labels_to_keep = [label for label, area in zip(properties['label'], properties['area']) 
                      if min_size <= area <= max_size]
    # Create a binary mask for objects to keep
    mask_to_keep = np.isin(mask, labels_to_keep)
    # Preserve original labels
    result_image = mask * mask_to_keep
    
    return result_image
