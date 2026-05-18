from cellpose import denoise
import torch


def Run_cellpose(input_image,model = "nuclei",model_restore = "denoise_nuclei",chan2 = False,
                 Diameter_size = 18,ch = [0,0]):
    """
    Run cellpose model
    Args:
        input_image: numpy array Depth * Height * Width
        model: str, default is "nuclei"
        model_restore: str, default is "denoise_nuclei"
        chan2: bool, default is False
        Diameter_size: int, default is 18
        ch: list, default is [0,0]
    Returns:    
        masks: numpy array
        flows: numpy array
        styles: numpy array
        imgs_dn: numpy
    """
    denoise_model = denoise.CellposeDenoiseModel(model_type=model,
                                                 restore_type=model_restore, 
                                                 chan2_restore=chan2, 
                                                 gpu=True,device=torch.device('cuda'))
    
    masks, flows, styles, imgs_dn = denoise_model.eval(input_image, 
                            channels=ch,do_3D=True,diameter=Diameter_size) 
    return masks, flows, styles, imgs_dn