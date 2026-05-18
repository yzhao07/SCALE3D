import skimage as sk
from skimage.measure import regionprops
import numpy as np
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import os

def compute_glcm_features(slice_):
    """
    Compute 2D GLCM texture features for one grayscale slice.

    Inputs:
        slice_ (ndarray): 2D image patch, typically uint8, used to build the
            gray-level co-occurrence matrix.

    Outputs:
        dict: Mapping from feature name to scalar value. Keys are
            'contrast', 'dissimilarity', 'homogeneity', 'energy',
            'correlation', and 'ASM'.
    """
    glcm = graycomatrix(slice_, distances=[1], angles=[0], levels=255, normed=True)
    return {prop: graycoprops(glcm, prop)[0, 0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']}

def compute_average_glcm_features_3planes(img, centroid,z1,x1,y1):
    """
    Compute GLCM features on XY, XZ, and YZ planes around one centroid.

    Inputs:
        img (ndarray): 3D grayscale image block containing the target nucleus.
        centroid (tuple or ndarray): Global centroid coordinates in z, x, y.
        z1 (int): Starting z index of the local block inside the full volume.
        x1 (int): Starting x index of the local block inside the full volume.
        y1 (int): Starting y index of the local block inside the full volume.

    Outputs:
        tuple: Five dictionaries in this order:
            1. avg_features
            2. std_features
            3. min_features
            4. max_features
            5. median_features
        Each dictionary contains the same GLCM keys prefixed with `glcm_`.
    """
    cz_global, cx_global, cy_global = centroid
    cz = int(cz_global - z1)
    cx = int(cx_global - x1)
    cy = int(cy_global - y1)

    # Extract the three orthogonal planes
    xy_plane = img[cz, :, :]
    xz_plane = img[:, cx, :]
    yz_plane = img[:, :, cy]

    # Compute GLCM features for each plane
    features_xy = compute_glcm_features(xy_plane)
    features_xz = compute_glcm_features(xz_plane)
    features_yz = compute_glcm_features(yz_plane)

    # Average the features
    avg_features = {}
    std_features = {}
    min_features = {}
    max_features = {}
    median_features = {}
    for key in features_xy:
        avg_features[f'glcm_{key}'] = np.mean([
            features_xy[key],
            features_xz[key],
            features_yz[key]])
        std_features[f'glcm_{key}'] = np.std([
            features_xy[key],
            features_xz[key],
            features_yz[key]])
        min_features[f'glcm_{key}'] = np.min([
            features_xy[key],
            features_xz[key],
            features_yz[key]])
        max_features[f'glcm_{key}'] = np.max([
            features_xy[key],
            features_xz[key],
            features_yz[key]])
        median_features[f'glcm_{key}'] = np.median([
            features_xy[key],
            features_xz[key],
            features_yz[key]])
        
    return avg_features,std_features,min_features,max_features,median_features

def calculate_surface_area(binary_mask):
    """
    Compute mesh-based surface area for one 3D binary object.

    Inputs:
        binary_mask (ndarray): 3D binary mask where the target object is 1 and
            background is 0.

    Outputs:
        float: Surface area estimated from the marching-cubes mesh.
    """
    verts, faces, _, _ = sk.measure.marching_cubes(binary_mask, level=0.5)
    surface_area = sk.measure.mesh_surface_area(verts, faces)
    return surface_area


def kdtree(centroids, k=10, distance_thres=np.inf):
    """
    Query nearest neighbors for centroid coordinates using `cKDTree`.

    Inputs:
        centroids (ndarray): Array of shape `(n, d)` containing centroid
            coordinates for `n` nuclei in `d` dimensions.
        k (int): Number of nearest neighbors to return per centroid.
        distance_thres (float): Optional maximum neighbor distance.

    Outputs:
        tuple: `(distances, indices)` where both arrays have shape `(n, k)`
        for normal cases. If fewer than 3 centroids are provided, both arrays
        are returned with shape `(n, 1)`.
    """
    n_centroids = len(centroids)
    
    if n_centroids < 3:
        return np.zeros((n_centroids, 1)), np.zeros((n_centroids, 1))
    # Build a k-d tree for efficient nearest neighbor search
    tree = cKDTree(centroids)
    # Query the k+1 nearest neighbors (including self)
    distances, indices = tree.query(centroids, k=k+1,distance_upper_bound=distance_thres)  # distances.shape = (n, k+1)
    return distances[:, 1:k+1],indices[:, 1:k+1]


def feature53_extraction_parallel(
    specimen_name,
    mask,
    input_data,
    cyto,
    gland_mask,
    n_thread=-1,
    k=15,
    chunk_size=500,
    temp_dir="./tmp/tmp",
):
    """
    Extract the 53-feature nuclei representation for one specimen block set.

    Inputs:
        specimen_name (str): Sample identifier used to name temporary memmap
            files.
        mask (ndarray): 3D labeled nuclei mask with one integer label per
            nucleus.
        input_data (ndarray): 3D nuclei image volume aligned with `mask`.
        cyto (ndarray): 3D cytoplasm image volume aligned with `mask`.
        gland_mask (ndarray): 3D gland-region mask aligned with `mask`.
        n_thread (int): Number of joblib workers. `-1` means use all cores.
        k (int): Number of nearest neighbors used for crowdedness features.
        chunk_size (int): Number of nuclei processed together per parallel job.
        temp_dir (str): Directory used to store temporary `.npy` memmap files.

    Outputs:
        tuple: `(node_label, node_feat)`
            node_label (ndarray): 1D array of valid nucleus ids.
            node_feat (ndarray): 2D float32 feature matrix with one row per
                nucleus and columns containing shape, texture, GLCM, and
                spatial features.
    """
    input_data = sk.exposure.rescale_intensity(input_data, in_range='image', out_range=(0, 255)).astype(np.uint8)
    cyto = sk.exposure.rescale_intensity(cyto, in_range='image', out_range=(0, 255)).astype(np.uint8)

    os.makedirs(temp_dir, exist_ok=True)
    print(os.path.join(temp_dir, f"{specimen_name}_nuclei_input.npy"))
    np.save(os.path.join(temp_dir, f"{specimen_name}_nuclei_input.npy"), input_data)
    np.save(os.path.join(temp_dir, f"{specimen_name}_cyto_input.npy"), cyto)
    # Reload as memory-mapped read-only
    input_data = np.load(os.path.join(temp_dir, f"{specimen_name}_nuclei_input.npy"), mmap_mode='r')
    cyto = np.load(os.path.join(temp_dir, f"{specimen_name}_cyto_input.npy"), mmap_mode='r')

    regions = regionprops(mask, cache=True)
    valid_regions = []
    for r in regions:
        try:
            if (
                r.area > 10 and
                r.convex_area > 10 and
                r.major_axis_length > 0 and
                r.minor_axis_length > 1):
                valid_regions.append(r)
        except (ValueError, AttributeError):
            continue

    centroids = np.array([r.centroid for r in valid_regions])
    distances, indices = kdtree(centroids, k=k)
    mean_crowdedness = np.mean(distances, axis=1)
    std_crowdedness = np.std(distances, axis=1)
    ball = sk.morphology.ball(10)
    print("start parallel")
    print(f"Using {n_thread} threads")
    print(f"Number of valid regions: {len(valid_regions)}")
    
    def process_region(i):
        try:
            print(f"Processing region {i}")
            r = valid_regions[i]
            region_id = r.label

            z1, x1, y1, z2, x2, y2 = r.bbox
            z1e, x1e, y1e = max(0, z1 - 10), max(0, x1 - 10), max(0, y1 - 10)
            z2e, x2e, y2e = min(mask.shape[0], z2 + 10), min(mask.shape[1], x2 + 10), min(mask.shape[2], y2 + 10)

            sg_mask = mask[z1:z2, x1:x2, y1:y2] == region_id
            sg_grey = input_data[z1:z2, x1:x2, y1:y2] * sg_mask
            sg_grey2 = np.clip(sg_grey.astype(np.int16), 1, 255) - 1
            sg_grey2 = sg_grey2.astype(np.uint8)

            sg_cyto = cyto[z1e:z2e, x1e:x2e, y1e:y2e]
            sg_cyto_mask = mask[z1e:z2e, x1e:x2e, y1e:y2e] == region_id
            sg_cyto_mask = sk.morphology.binary_dilation(sg_cyto_mask, ball)

            surface_area = calculate_surface_area(sg_mask)
            aspect_ratio = r.major_axis_length / r.minor_axis_length
            roughness = surface_area / r.area
            sphericity = (np.pi ** (1 / 3)) * ((6 * r.area) ** (2 / 3)) / surface_area

            feats_shape = [
                r.area, r.convex_area, r.equivalent_diameter, r.extent, r.filled_area,
                r.major_axis_length, r.minor_axis_length, r.solidity,
                surface_area, aspect_ratio, roughness, sphericity]

            feats_texture = [
                np.std(sg_grey[sg_mask]), np.mean(sg_grey[sg_mask]),
                np.min(sg_grey[sg_mask]), np.max(sg_grey[sg_mask]),
                np.std(sg_cyto[sg_cyto_mask]), np.mean(sg_cyto[sg_cyto_mask]),
                np.min(sg_cyto[sg_cyto_mask]), np.max(sg_cyto[sg_cyto_mask]),
                mean_crowdedness[i], std_crowdedness[i]]

            cz, cx, cy = map(int, r.centroid)
            gland_pos = gland_mask[cz, cx, cy]
            feats_spatial = [gland_pos, cz, cx, cy]
            # intensity feat
            avg, std, minf, maxf, med = compute_average_glcm_features_3planes(sg_grey2, r.centroid,z1,x1,y1) #(img, centroid,z1,x1,y1)
            hist = np.histogram(sg_grey2[sg_mask], bins=32, density=True)[0] + 1e-10
            nucleus_intensity_feats = [ entropy(hist)] + [avg[k] for k in avg] + [std[k] for k in std] + [minf[k] for k in minf] + [maxf[k] for k in maxf] + [med[k] for k in med]
            all_feat = np.hstack(feats_shape + feats_texture + nucleus_intensity_feats+ feats_spatial)
            print(region_id,all_feat.shape)
            return region_id, all_feat

        except Exception as e:
            print(f"[ERROR] region {i}: {e}")
            return None
    
    # Chunk and parallel
    chunks = [list(range(i, min(i + chunk_size, len(valid_regions)))) for i in range(0, len(valid_regions), chunk_size)]
    def process_chunk(chunk_idx):
        out = []
        for i in chunk_idx:
            res = process_region(i)
            if res is not None:
                out.append(res)
        return out

    results = Parallel(n_jobs=n_thread)(delayed(process_chunk)(chunk) for chunk in chunks) #,backend='threading'
    flat = [item for sub in results for item in sub]
    if len(flat) == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    node_label = np.array([rid for (rid, _) in flat], dtype=np.int32)
    node_feat  = np.vstack([fv for (_, fv) in flat]).astype(np.float32, copy=False)

    return node_label, node_feat
