import os
import time
import argparse
import nibabel as nib
import numpy as np
from utils.feature_extraction import feature53_extraction_parallel
from utils.postprocess import postprocess
from utils.utils import read_niigz, readin_h5

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def _default_sample_name(path):
    parent_name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent_name:
        return parent_name
    return os.path.basename(os.path.abspath(path))


def main(): 
    parser = argparse.ArgumentParser(description="Perform feature extraction & adjacency matrix for a given cellpose segmentation path")
    parser.add_argument('--h5', type=str, required=True, help="h5")
    parser.add_argument('--CellposeDir', type=str, required=True, help="cellpose directory")
    parser.add_argument('--gland', type=str, required=True, help="gland mask")
    parser.add_argument('--startIdx', type=int, default=0,help="start idx")
    parser.add_argument('--nthreads', type=int, default=30,help="CUP threads")
    parser.add_argument('--outdir', type=str, default=None, help="output directory for individual cell features")
    parser.add_argument('--sample_name', type=str, default=None, help="optional sample name for logging/temp files")
    parser.add_argument('--block_size', type=int, default=1024, help="XY block size")
    parser.add_argument('--grid_size', type=int, default=4, help="number of blocks per axis")
    parser.add_argument('--depth', type=int, default=512, help="maximum depth to process")
    parser.add_argument('--cyto_channel', type=str, default='s01', help="cytoplasm channel")
    parser.add_argument('--temp_dir', type=str, default="./tmp/tmp", help="temporary array directory")
    args = parser.parse_args()

    
    start_time = time.time()
    cellpose_dir = args.CellposeDir
    startidx = args.startIdx
    sample_name = args.sample_name or _default_sample_name(args.CellposeDir)
    print("parts:", sample_name, args.CellposeDir, args.gland)
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.CellposeDir)), "individual_cells")
    os.makedirs(outdir, exist_ok=True)
    
    print("start feature extraction for ", args.CellposeDir)
    gland_mask = nib.load(args.gland)
    depth_axis = int(np.argmin(gland_mask.shape))
    depth = min(args.depth, gland_mask.shape[depth_axis])
    print('depth:',depth)

    for x in range(args.grid_size):
        for y in range(args.grid_size):
            start_time1 = time.time()
            feat_name = "feature53_blk_"+str(x)+"_"+str(y)+".npz"
            feat_label_name = "label53_blk_"+str(x)+"_"+str(y)+".npz"
            if os.path.exists(os.path.join(outdir,feat_name))  and os.path.exists(os.path.join(outdir,feat_label_name)):
                print("already processed")
                continue
            
            mask_file = "mask_blk_"+str(x)+"_"+str(y)+".nii.gz"
            data_file= "imgdn_blk_"+str(x)+"_"+str(y)+".nii.gz"
            print(mask_file)
            ###### load data
            mask = read_niigz(os.path.join(cellpose_dir,mask_file)).astype(int)
            input_data = read_niigz(os.path.join(cellpose_dir,data_file))
            if depth_axis == 0:
                gland_blk = gland_mask.dataobj[
                    startidx:(startidx + depth),
                    (x * args.block_size):((x + 1) * args.block_size),
                    (y * args.block_size):((y + 1) * args.block_size),
                ]
            elif depth_axis == 2:
                gland_blk = gland_mask.dataobj[
                    (x * args.block_size):((x + 1) * args.block_size),
                    (y * args.block_size):((y + 1) * args.block_size),
                    startidx:(startidx + depth),
                ]
                gland_blk= np.transpose(gland_blk, (2, 0, 1)) #z,x,y
            else:
                print("gland mask wrong")
                continue
            
            cyto = readin_h5(
                args.h5,
                "0", #original resolution
                args.cyto_channel,
                startidx,
                (startidx + depth),
                (x * args.block_size),
                ((x + 1) * args.block_size),
                (y * args.block_size),
                ((y + 1) * args.block_size),
            )
            mask = mask[0:depth]
            input_data=input_data[0:depth]
            print("input shape:",mask.shape,input_data.shape,gland_blk.shape)
            
            ###### postprocessing
            mask=postprocess(mask,min_size=150,max_size=np.inf)
            print("mask shape after postprocess:",mask.shape)
            ###### feature matrix mask, input_data, cyto,gland_mask,n_thread
            node_label,node_feat = feature53_extraction_parallel(
                sample_name,
                mask,
                input_data,
                cyto,
                gland_blk,
                n_thread=args.nthreads,
                temp_dir=args.temp_dir,
            )
            
            np.savez_compressed(os.path.join(outdir,feat_name), node_feat)
            np.savez_compressed(os.path.join(outdir,feat_label_name), node_label)
            end_time = time.time()
            elapsed_time = end_time - start_time1
            print(f"block in Elapsed time: {elapsed_time:.2f} seconds")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"parallel processing in Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
