import os
import time
import argparse
import joblib
import numpy as np
from supercell_graph.cell_graph import supernode_formation_region_grow_similarity


def _default_sample_name(path):
    parent_name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent_name:
        return parent_name
    return os.path.basename(os.path.abspath(path))


def main():
    parser = argparse.ArgumentParser(description="Perform supercell identification based on cell segmentation")
    parser.add_argument('--cellseg', type=str, required=True, help="segmentation directory")
    parser.add_argument('--patch_size', type=int, default=1024, help="patch size")
    parser.add_argument('--max_num', type=int, default=15, help="maximum number of cells")
    parser.add_argument('--distance_thres', type=int, default=30, help="distance for radius")
    parser.add_argument('--similarity', type=float, default=0.7, help="similarity threshold")
    parser.add_argument('--normalization',dest='normalization',type=str,default="../Evaluation/test/feat53_mean_std_scaler.pkl",required=True,help="normalization scaler pickle path")
    parser.add_argument('--sample_name', type=str, default=None, help="optional sample name")
    parser.add_argument('--individual_cells_dir', type=str, default=None, help="directory of individual cell features")
    parser.add_argument('--out_folder', type=str, default=None, help="output directory for supercell features")
    parser.add_argument('--grid_size', type=int, default=4, help="number of blocks per axis")
    parser.add_argument('--block_size', type=int, default=1024, help="XY block size used for individual cell features")
    parser.add_argument('--slide_size', type=int, default=4096, help="full XY size of the slide/volume")
    parser.add_argument('--feature_dim', type=int, default=53, help="number of per-cell features before metadata columns")
    parser.add_argument('--min_size', type=int, default=3, help="minimum cells per supernode")
    parser.add_argument('--min_cells_per_type', type=int, default=100, help="minimum epithelial/stromal cells to cluster")
    parser.add_argument('--epithelium_label', type=int, default=1, help="epithelium label in the input features")
    parser.add_argument('--stroma_label', type=int, default=0, help="stroma label in the input features")

    args = parser.parse_args()

    
    print(args.cellseg)
    print("patch_size",args.patch_size)
    rangex = int(args.slide_size / args.patch_size)
    rangey = int(args.slide_size / args.patch_size)
    max_num = args.max_num
    distance_thres = args.distance_thres
    sim = args.similarity
    
    total_num = args.feature_dim + 5
    scaler = joblib.load(args.normalization)
    sample_name = args.sample_name or _default_sample_name(args.cellseg)
    selected_cols = list(range(args.feature_dim))
    result_folder = args.individual_cells_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.cellseg)),
        'individual_cells',
    )
    out_folder = args.out_folder or os.path.join(
        os.path.dirname(os.path.abspath(args.cellseg)),
        'supercell',
    )
    print("output folder: ", out_folder)
    os.makedirs(out_folder, exist_ok=True)
    expected_feature_paths = [
        os.path.join(result_folder, f"feature53_blk_{x}_{y}.npz")
        for x in range(args.grid_size)
        for y in range(args.grid_size)
    ]
    if any(os.path.exists(path) for path in expected_feature_paths):
        print("start processing: ", result_folder)
        start_time = time.time()
        X_all_blk = []
        for x in range(args.grid_size):
            for y in range(args.grid_size):
                fileshape= "feature53_blk_"+str(x)+"_"+str(y)+".npz"
                nuclei_id_label = "label53_blk_"+str(x)+"_"+str(y)+".npz"
                feature_path = os.path.join(result_folder, fileshape)
                label_path = os.path.join(result_folder, nuclei_id_label)
                if not (os.path.exists(feature_path) and os.path.exists(label_path)):
                    print("missing block:", fileshape)
                    continue
                data_shape = np.load(feature_path)["arr_0"]
                nuclei_id_label = np.load(label_path)["arr_0"]
                if data_shape.shape[0] == 0:
                    print("empty block:", fileshape)
                    continue
                X = np.concatenate([
                    data_shape,
                    nuclei_id_label.reshape(-1, 1) # (N, 1) nuclei id label
                ], axis=1)

                print("x range", np.min(X[:,-3]), np.max(X[:,-3]),
                      "y range", np.min(X[:,-2]), np.max(X[:,-2]),"z range", np.min(X[:,-4]), np.max(X[:,-4]),
                      "cell type", np.unique(X[:,-5], return_counts=True) )

                assert X.shape[1] == total_num
                X[:,-3] += x * args.block_size
                X[:,-2] += y * args.block_size
                X_all_blk.append(X)
        if not X_all_blk:
            print("skip: no individual cell feature blocks found")
            return
        X_all_blk = np.concatenate(X_all_blk, axis=0)
        X_all_blk[:, -4:-1] = X_all_blk[:, -4:-1].astype(int)
        print("all blk combined: ", X_all_blk.shape)
        assert X_all_blk.shape[1]==total_num
        s = str(sim).replace(".", "")
        for x in range(rangex):
            for y in range(rangey):
                supernode_feat = "supercell_"+str(args.patch_size)+f"_feature{args.feature_dim}_d{distance_thres}_blk_"+str(x)+"_"+str(y)+".npz"
                supernode_nuclei_id_list = "supercell_"+str(args.patch_size)+f"_nucleiId{args.feature_dim}_d{distance_thres}_blk_"+str(x)+"_"+str(y)
                if os.path.exists(os.path.join(out_folder,supernode_feat)):
                    print("skip: ", supernode_feat)
                    continue
                xstart = x*args.patch_size
                xend = (x+1)*args.patch_size
                ystart = y*args.patch_size
                yend = (y+1)*args.patch_size
                X_subset = X_all_blk[(X_all_blk[:, -3] >= xstart) & 
                                        (X_all_blk[:, -3] < xend) &
                                        (X_all_blk[:, -2] >= ystart) & 
                                        (X_all_blk[:, -2] < yend)]
                if X_subset.shape[0] == 0:
                    print("skip empty patch:", x, y)
                    continue
                X_subset[:,-3] -= xstart
                X_subset[:,-2] -= ystart
                ### stroma & epi
                X_epithelium = X_subset[X_subset[:,-5] == args.epithelium_label]
                X_stroma = X_subset[X_subset[:,-5] == args.stroma_label]
                print(x,y,"total num of epi and stroma",X_epithelium.shape, X_stroma.shape)
                
                if X_epithelium.shape[0] < args.min_cells_per_type:
                    X_epithelium = np.empty((0, total_num))
                    epi_supernode_labels = np.array([], dtype=int)
                    epi_supernode_features = np.empty((0, ((args.feature_dim) * 5 + 1)))
                    epi_supernode_centroids = np.empty((0, 4)) 
                    epi_supernode_to_nuclei_ids = {}
                else:
                    epi_nuclei_ids = X_epithelium[:, -1].astype(int)
                    epi_supernode_labels, epi_supernode_features,epi_supernode_centroids,epi_supernode_to_nuclei_ids = supernode_formation_region_grow_similarity(scaler, 
                                X_epithelium[:,selected_cols], X_epithelium[:,-4:-1],epi_nuclei_ids,
                                similarity_thres= sim, max_num=max_num, max_distance = distance_thres,min_size=args.min_size,filter=True)
                    epi_cell_type = np.ones((epi_supernode_centroids.shape[0], 1)) # epithelium as 1
                    epi_supernode_centroids = np.hstack([epi_supernode_centroids, epi_cell_type]).astype(int)
                
                if X_stroma.shape[0] < args.min_cells_per_type:
                    X_stroma = np.empty((0, total_num))
                    stroma_supernode_labels = np.array([], dtype=int)
                    stroma_supernode_features = np.empty((0, ((args.feature_dim) * 5 + 1)))
                    stroma_supernode_centroids = np.empty((0, 4))
                    stroma_supernode_to_nuclei_ids = {}
                else:
                    stroma_nuclei_ids = X_stroma[:, -1].astype(int)
                    stroma_supernode_labels, stroma_supernode_features,stroma_supernode_centroids,stroma_supernode_to_nuclei_ids = supernode_formation_region_grow_similarity(scaler, 
                        X_stroma[:,selected_cols],X_stroma[:,-4:-1],stroma_nuclei_ids,
                        similarity_thres= sim, max_num=max_num, max_distance = distance_thres,min_size=args.min_size,filter=True)
                    stroma_cell_type = np.zeros((stroma_supernode_centroids.shape[0], 1)) # stroma as 0
                    stroma_supernode_centroids = np.hstack([stroma_supernode_centroids, stroma_cell_type]).astype(int)

                assert stroma_supernode_features.shape[1] == epi_supernode_features.shape[1]
                supernode_centroids = np.vstack([epi_supernode_centroids, stroma_supernode_centroids])
                supernode_features = np.vstack([epi_supernode_features, stroma_supernode_features])
                super_feat = np.hstack((supernode_features, supernode_centroids))
                offset = max(epi_supernode_to_nuclei_ids.keys(), default=-1) + 1
                dict2_shifted = {k + offset: v for k, v in stroma_supernode_to_nuclei_ids.items()}
                merged_dict = {**epi_supernode_to_nuclei_ids, **dict2_shifted}
                np.savez_compressed(os.path.join(out_folder,supernode_feat), super_feat)
                np.save(os.path.join(out_folder,supernode_nuclei_id_list), merged_dict, allow_pickle=True)

        end_time= time.time()
        print("time: ", end_time-start_time)      
    else:
        print("skip: ", args.cellseg)

if __name__ == "__main__":
    main()
