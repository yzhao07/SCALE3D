import h5py as h5
import nibabel as nib

def readin_h5(h5path,res,chan,s,e,x1,x2,y1,y2):
    """
    Read in .h5 file
    Args:               
        h5path: str
        res: str, 0 - original, 1 - downsampled
        chan: str, "s00" - nuclei, "s01" - cyto except for EAM001 and otsl5
        s: int
        e: int
        x1: int
        x2: int
        y1: int
        y2: int
    Returns:
        img: numpy array
    """
    print("reading .h5 file...",s,e)
    with h5.File(h5path, 'r') as f:
        img = f['t00000'][chan][res]['cells'][s:e,x1:x2,y1:y2]
    return img


def read_niigz(filepath):
    """
    Reads a NIfTI (.nii.gz) file and returns the image data.
    Parameters:
    filepath (str): The path to the .nii.gz file to be read.
    Returns:
    numpy.ndarray: The image data contained in the NIfTI file.
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    return data
