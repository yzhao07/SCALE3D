import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
import time
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
import cupy as cp
import rapids_singlecell as rsc
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

def process_one_specimen(
    specimen_name,
    data_name,
    supercell_root,
    block_size,
    epithelial_or_stromal=1,
    grid_size=4,
):

    outdir = os.path.join(supercell_root, specimen_name, "supercell")
    if not os.path.exists(outdir):
        return None
    collected = []
    ids = []

    for x in range(grid_size):
        for y in range(grid_size):
            path = os.path.join(outdir, f"{data_name}_{x}_{y}.npz")
            if not os.path.exists(path):
                continue

            X = np.load(path)["arr_0"]
            if X.shape[0] == 0:
                continue

            # shift coordinates (z, x, y, celltype are last 4 columns)
            X[:, -3] = X[:, -3].astype(int) + x * block_size
            X[:, -2] = X[:, -2].astype(int) + y * block_size
                        
            mask = X[:, -1] == epithelial_or_stromal
            X_sel = X[mask]
            if X_sel.shape[0] == 0:
                continue

            collected.append(X_sel)
            ids.extend([specimen_name] * X_sel.shape[0])

    if len(collected) == 0:
        return None

    return np.concatenate(collected, axis=0), np.array(ids)


def main():
    parser = argparse.ArgumentParser(description="Perform clustering for super cells based on extracted features")
    parser.add_argument('--resolution', type=float,help="resolution for leiden clustering", default=0.1)
    parser.add_argument('--file_list', type=str, help="file_list containing name, directory")
    parser.add_argument('--data', type=str, required=True, help="dataname")
    parser.add_argument('--neighbors', type=int, default=50, help="n_neighbors for graph")
    parser.add_argument("--outfile",type=str, required=True, help="outfile name for h5ad")
    parser.add_argument("--epithelial_or_stromal", type=int,required=True,help=("1 for epithelial, 0 for stromal"))
    parser.add_argument("--supercell_root", type=str, required=True, help="root directory containing <sample>/supercell folders")
    parser.add_argument("--sample_column", type=str, default="name", help="sample column in the sample list CSV")
    parser.add_argument("--n_jobs", type=int, default=16, help="parallel specimen workers")
    parser.add_argument("--block_size", type=int, default=1024, help="XY block size")
    parser.add_argument("--grid_size", type=int, default=4, help="number of blocks per axis")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id for RAPIDS")
    
    args = parser.parse_args()

    res = args.resolution
    nn = args.neighbors

    start = time.time()
    data_name = args.data
    print(f'clustering with resolution {res}; ',data_name)
    all_x = []
    specimen = []
    file_list = pd.read_csv(args.file_list)
    if args.sample_column != "name":
        file_list = file_list.rename(columns={args.sample_column: "name"})
    if "name" not in file_list.columns:
        raise ValueError("sample list CSV must contain a sample column")
    print(file_list["name"],"total sample",file_list.shape)
    results = Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(process_one_specimen)(
            name,
            data_name,
            args.supercell_root,
            args.block_size,
            args.epithelial_or_stromal,
            args.grid_size,
        )
        for name in file_list["name"]
    )
    
    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("no supercell features were found for the requested cohort")
    all_x_list, specimen_list = zip(*results)
    all_x = np.concatenate(all_x_list, axis=0)
    specimen = np.concatenate(specimen_list, axis=0)

    print("Final:", all_x.shape, specimen.shape)
    feat = all_x[:, 1:54]
    adata = sc.AnnData(feat)  # X is your (n_cells x n_features) feature matrix
    adata.X = adata.X.astype(np.float32)
    adata.obs['cell_type'] = all_x[:,-1]
    adata.obsm['centroids'] = all_x[:,-4:-1]
    adata.obs['slide_id'] = specimen
    t = time.time() - start
    
    cp.cuda.Device(args.gpu_id).use()
    print(rmm.mr.get_current_device_resource_type())
    rmm.reinitialize(
        managed_memory=False,
        pool_allocator=True,
        devices=args.gpu_id,
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)
    print(f"RAPIDS memory manager initialized on GPU {cp.cuda.Device().id}")
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.scale(adata)
    print(adata)
    rsc.pp.pca(adata, n_comps=20)
    print(adata)
    rsc.pp.harmony_integrate(adata, key="slide_id", dtype=cp.float32)
    # -----------------------------
    #  Graph + Leiden + UMAP
    # -----------------------------
    # neighbors 
    rsc.pp.neighbors(adata, n_neighbors=nn, n_pcs=20,use_rep="X_pca_harmony",key_added="harmony")
    rsc.tl.umap(adata,neighbors_key="harmony",key_added="X_umap_harmony")
    rsc.tl.leiden(adata, resolution=res,neighbors_key="harmony",key_added="leiden_harmony")
    
    t = time.time() - start
    print(f"umap {t:.2f} seconds")
    rsc.get.anndata_to_CPU(adata)
    output_path = args.outfile if args.outfile.endswith(".h5ad") else f"{args.outfile}.h5ad"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    adata.write(output_path)
    t = time.time() - start
    print(f"[TIMER] whole pipeline took {t:.2f} seconds")



if __name__ == "__main__":
    main()
