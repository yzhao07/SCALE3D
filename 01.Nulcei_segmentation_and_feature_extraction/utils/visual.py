import numpy as np
import os
import skimage as sk
from skimage.segmentation import find_boundaries
from skimage import transform
import cv2


def visualize_mask_as_video(output_mask,input_data, dir,outavi,downsample=True):
    """
    Visualize a 3D mask as a video with optional downsampling.
    Parameters:
        output_mask (numpy.ndarray): 3D array of mask labels.
        input_data (numpy.ndarray): 3D array of input data to be visualized.
        dir (str): Directory to save the output video.
        outavi (str): Name of the output video file.
        downsample (bool, optional): If True, downsample the video. Default is True.
    Returns:
        None
    """
    ####### visual convert to rgb
    input_data = sk.exposure.rescale_intensity(input_data, in_range='image', out_range=(0, 255)).astype(np.uint8)
    num_labels = int(np.max(output_mask)+1) # Number of unique labels
    color_map = np.random.randint(0, 256, size=(num_labels, 3), dtype=np.uint8) 
    color_map[0] = [0, 0, 0]
    rgba_volume = color_map[output_mask]
    
    alpha = 0.5  # Alpha for blending
    #alpha = np.clip(input_data[..., None], 0.3, 1.0) 
    #rgba_volume = (rgba_volume * alpha + (1 - alpha) * np.array([0, 0, 0])).astype(np.uint8)  # Blend with alpha
    rgba_volume = (rgba_volume * alpha + (1 - alpha) * input_data[..., None]).astype(np.uint8) 
    # background
    background_indices = (output_mask == 0)
    #grayscale_255 = (input_data[background_indices] * 255).astype(np.uint8)
    grayscale_255 = (input_data[background_indices]).astype(np.uint8)# Scale grayscale to 0-255
    rgba_volume[background_indices] = np.stack([grayscale_255] * 3, axis=-1)  # Replace background with grayscale RGB
    
    for z in range(output_mask.shape[0]):
        boundaries = find_boundaries(output_mask[z], mode='outer')
        rgba_volume[z][boundaries]=[0.0,255.0,0.0]

    ##########Visualization
    # if downsample
    if downsample:
        data = transform.downscale_local_mean(rgba_volume, # comment this if don't want downsampling
                                                (1,2, 2, 1))
    else:
        print("original resolution")
        data = rgba_volume
        
    data = data.astype(np.uint8)
    depth,height, width,_= data.shape
    # output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG codec
    out = cv2.VideoWriter(os.path.join(dir,outavi), fourcc, 20, (width, height))
    for i in range(depth):
        slice_img = data[i]
        colored_img = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
        out.write(colored_img)

    out.release()


def visualize_imgdn_as_video(input_data, dir, outavi, downsample=True):
    """
    Visualizes a 3D image dataset as a video.
    Parameters:
    -----------
    input_data : numpy.ndarray
        The input 3D image data to be visualized. Expected to be in grayscale.
    dir : str
        The directory where the output video file will be saved.
    outavi : str
        The name of the output video file.
    downsample : bool, optional
        If True, the image data will be downsampled by a factor of 2 in both width and height. 
        Default is True.
    Returns:
    --------
    None
    """
    input_data = sk.exposure.rescale_intensity(input_data, in_range='image', out_range=(0, 255)).astype(np.uint8) 
    rgba_volume = np.stack([input_data] * 3, axis=-1) # Convert grayscale to RGB
    if downsample:
        data = transform.downscale_local_mean(rgba_volume, (1,2, 2, 1))
    else:
        print("original resolution")
        data = rgba_volume
    
    data = data.astype(np.uint8)
    depth,height, width,_= data.shape
    # output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG codec
    out = cv2.VideoWriter(os.path.join(dir,outavi), fourcc, 20, (width, height))
    for i in range(depth):
        slice_img = data[i]
        colored_img = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
        out.write(colored_img)

    out.release()
