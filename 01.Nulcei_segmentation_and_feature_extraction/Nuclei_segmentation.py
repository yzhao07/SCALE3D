## nuclei segmentation
import os
import time
import argparse
import nibabel as nib
import numpy as np
from model.Cellpose import Run_cellpose
from utils.preprocess import PreprocessNucleiChannel
from utils.utils import readin_h5
from utils.visual import visualize_mask_as_video, visualize_imgdn_as_video
from cellpose import core

def _default_sample_name(h5_path):
    parent_name = os.path.basename(os.path.dirname(os.path.abspath(h5_path)))
    if parent_name:
        return parent_name
    return os.path.splitext(os.path.basename(h5_path))[0]


def main():
    parser = argparse.ArgumentParser(description="Perform cellpose nuclei segmentation for a given h5 image")
    parser.add_argument('--h5', type=str, required=True, help="h5 image")
    parser.add_argument('--startIdx', type=int, default=0,help="start idx")
    parser.add_argument('--depth', type=int, default=600,help="how many depth to process")
    parser.add_argument('--nuclei_channel', type=str, default='s00',help="nuclei channel")
    parser.add_argument('--median', type=int, default=2,help="median kernel size")
    parser.add_argument('--diameter', type=int, default=18,help="estimated diameter size")
    parser.add_argument('--visualization', type=lambda x: x.lower() == 'true', default=True,help="if visualize")
    parser.add_argument('--outdir', type=str, default=None, help="output directory for cellpose results")
    parser.add_argument('--sample_name', type=str, default=None, help="optional sample name for logging")
    parser.add_argument('--block_size', type=int, default=1024, help="XY block size")
    parser.add_argument('--grid_size', type=int, default=4, help="number of blocks per axis")
    parser.add_argument('--gpu_id', type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d'%use_GPU)
    start_time = time.time()
    affine = np.eye(4)

    sample_name = args.sample_name or _default_sample_name(args.h5)
    outdir = args.outdir or os.path.join(os.path.dirname(args.h5), "cellpose")
    print(outdir)
    os.makedirs(outdir, exist_ok=True)
   
    ######## read in h5 image
    for x in range(args.grid_size):
        for y in range(args.grid_size):
            file_mask = 'mask_blk_'+str(x)+"_"+str(y)+'.nii.gz'
            file_imgdn= 'imgdn_blk_'+str(x)+"_"+str(y)+'.nii.gz'
            if os.path.exists(os.path.join(outdir,file_mask)) and os.path.exists(os.path.join(outdir,file_imgdn)):
                print("skip processing...", sample_name, "blk_", x, "_", y)
                continue

            print("processing...blk_", x, "_", y)
            img = readin_h5(
                args.h5,
                '0',
                args.nuclei_channel,
                args.startIdx,
                args.startIdx + args.depth,
                x * args.block_size,
                (x + 1) * args.block_size,
                y * args.block_size,
                (y + 1) * args.block_size,
            )
            img = PreprocessNucleiChannel(img, median_kernel=args.median)
            masks, flows, styles, imgs_dn = Run_cellpose(img, Diameter_size=args.diameter)
            print(
                masks.shape,
                imgs_dn.shape,
                masks.dtype,
                imgs_dn.dtype,
                np.max(imgs_dn),
                np.min(imgs_dn),
                np.max(masks),
            )

            img_mask = nib.Nifti1Image(masks.squeeze(), affine)
            nib.save(img_mask, os.path.join(outdir, file_mask))

            img_dn = nib.Nifti1Image(imgs_dn.squeeze(), affine)
            nib.save(img_dn, os.path.join(outdir, file_imgdn))
            print("Done processing...blk_", x, "_", y)

            if args.visualization:
                print("Visualizing...")
                visualize_mask_as_video(masks.squeeze(),imgs_dn.squeeze(),outdir,'mask_blk_'+str(x)+"_"+str(y)+'.avi')
                print("Visualizing...")
                visualize_imgdn_as_video(imgs_dn.squeeze(),outdir,'imgdn_blk_'+str(x)+"_"+str(y)+'.avi')
                
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"original Elapsed time: {elapsed_time:.2f} seconds")
    
if __name__ == "__main__":
    main()
