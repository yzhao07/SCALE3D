import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from scipy.spatial import cKDTree

def supernode_formation_region_grow_similarity(
    scaler,
    X_data: np.ndarray,
    centroids: np.ndarray,
    nuclei_ids: np.ndarray,
    similarity_thres: float = 0.7,   # cosine threshold in [0,1]
    max_num: int = 30,               # max cells per supernode
    max_distance: float = 30,        # radius (px/vox) + seed-ball cap
    min_size: int = 3,
    filter: bool = False,            # keep name for backward compat
    sort_neighbors: bool = True,
):
    """
    Returns:
        supernode_labels: (N,) np.int32 cluster labels per node (-1 for unclustered)
        supernode_features: (K, 1+5F) features per supernode [count, mean, min, max, std, median]
        supernode_centroids: (K, D) spatial centroid per supernode
        supernode_to_nuclei_ids: dict[row_index -> list of nuclei_ids]
    """
    # ---- scale ----
    X_scale = scaler.transform(X_data).astype(np.float32, copy=False)
    n, feature_dim = X_scale.shape
    print("within supernode formation: feat shape", feature_dim)

    tau_cos = float(similarity_thres)

    # ---- radius neighbors ----
    r = float(max_distance)
    tree = cKDTree(centroids)
    balls = tree.query_ball_point(centroids, r=r)  # includes self
    if sort_neighbors:
        #balls = [sorted(lst) for lst in balls]     # deterministic neighbor order
        balls = [
            sorted(lst, key=lambda j: np.sum((centroids[j] - centroids[i])**2))
            for i, lst in enumerate(balls)
        ]

    supernode_labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)

    label = 0
    seed_xy_list = []  # keep seed position per cluster label (for attach cap)

    # RNG / seed order
    rng = np.random.default_rng(42)
    seed_order = np.arange(n) if rng is None else rng.permutation(n)

    # ---- region grow with seed-ball cap (cosine gate) ----
    for seed in seed_order:
        if visited[seed]:
            continue

        cluster = [seed]
        visited[seed] = True

        mu_z = X_scale[seed].copy()       # running centroid vector in feature space
        seed_xy = centroids[seed].copy()  # fixed anchor for spatial cap

        frontier = deque(v for v in balls[seed] if v != seed)
        in_frontier = np.zeros(n, dtype=bool)
        if len(frontier) > 0:
            in_frontier[np.array(list(frontier))] = True

        while frontier and len(cluster) < max_num:
            current = frontier.popleft()   # BFS for radial balance
            in_frontier[current] = False
            if visited[current]:
                continue

            # (A) seed-ball constraint → keep cluster spatially bounded
            if np.linalg.norm(centroids[current] - seed_xy) > r:
                continue

            # (B) cosine gate vs current centroid vector
            cval = float(cosine_similarity(
                X_scale[current:current+1],          # shape (1, F)
                mu_z.reshape(1, -1)                  # shape (1, F)
            )[0, 0])

            if cval < tau_cos:
                continue

            # accept
            cluster.append(current)
            visited[current] = True
            # update centroid (simple running mean vector)
            mu_z = (mu_z * (len(cluster)-1) + X_scale[current]) / len(cluster)

            # expand frontier
            for nb in balls[current]:
                if nb == current or visited[nb] or in_frontier[nb]:
                    continue
                frontier.append(nb)
                in_frontier[nb] = True

            if len(cluster) >= max_num:
                break

        if len(cluster) >= min_size:
            supernode_labels[cluster] = label
            seed_xy_list.append(seed_xy)   # store seed anchor for this label
            label += 1
        # else leave them unclustered (-1)

    # ---- post-attach: assign leftovers if within caps (cosine gate) ----
    unassigned = np.where(supernode_labels < 0)[0]
    if unassigned.size > 0 and label > 0:
        # per-cluster stats
        sizes  = np.zeros(label, dtype=int)
        mu_z_c = np.zeros((label, feature_dim), dtype=np.float32)  # centroid vectors
        mu_xy  = np.zeros((label, centroids.shape[1]), dtype=np.float32)
        for lab in range(label):
            idx = np.where(supernode_labels == lab)[0]
            sizes[lab] = idx.size
            mu_z_c[lab] = X_scale[idx].mean(axis=0)
            mu_xy[lab]  = centroids[idx].mean(axis=0)

        open_mask = sizes < max_num
        open_ids  = np.where(open_mask)[0]
        attached = 0

        for u in unassigned:
            if open_ids.size == 0:
                break
            # spatial: cluster centroid proximity within r
            d_sp = np.linalg.norm(mu_xy[open_ids] - centroids[u], axis=1)
            cand = open_ids[d_sp <= r]
            if cand.size == 0:
                continue

            # feature: cosine to cluster centroid vector
            cosines = cosine_similarity(
                X_scale[u:u+1],                      # (1, F)
                mu_z_c[cand]                         # (C, F)
            ).ravel()                                # (C,)
            m = int(np.argmax(cosines))
            best_lab = int(cand[m])
            best_cos = float(cosines[m])

            # enforce seed-ball constraint too (hard bound)
            if np.linalg.norm(centroids[u] - seed_xy_list[best_lab]) > r:
                continue
            if (best_cos >= tau_cos) and (sizes[best_lab] < max_num):
                n_now = sizes[best_lab]
                xu = X_scale[u]
                # update centroids
                mu_z_c[best_lab] = (mu_z_c[best_lab] * n_now + xu) / (n_now + 1)
                mu_xy[best_lab]  = (mu_xy[best_lab]  * n_now + centroids[u]) / (n_now + 1)
                sizes[best_lab] += 1
                supernode_labels[u] = best_lab
                attached += 1
                if sizes[best_lab] >= max_num:
                    open_mask[best_lab] = False
                    open_ids = np.where(open_mask)[0]
        print(f"[INFO] Post-attach: assigned {attached} / {unassigned.size} previously unclustered nodes")

    unassigned_idx = np.where(supernode_labels == -1)[0]
    print(f"[INFO] Nodes left unclustered: {len(unassigned_idx)} / {n} ({len(unassigned_idx)/n:.2%})")

    X_used = X_data
    C_used = centroids
    L_used = supernode_labels
    N_used = nuclei_ids
    if filter:
        clustered_mask = supernode_labels >= 0
        X_used = X_data[clustered_mask]
        C_used = centroids[clustered_mask]
        L_used = supernode_labels[clustered_mask]
        N_used = nuclei_ids[clustered_mask]

    valid_labels = np.unique(L_used[L_used >= 0])
    supernode_feature_list = []
    supernode_centroid_list = []
    supernode_to_nuclei_ids = {}
    count_all = []
    count = 0
    N_used = nuclei_ids if not filter else nuclei_ids[supernode_labels >= 0]
    cluster_max_radii = []

    for lab in valid_labels:
        node_indices = np.where(L_used == lab)[0]
        features = X_used[node_indices]
        pts = C_used[node_indices]
        centroid = C_used[node_indices].mean(axis=0)

        # cluster "radius" (max distance to centroid in px)
        if pts.shape[0] > 0:
            d = np.linalg.norm(pts - centroid, axis=1)
            max_center_dist = float(d.max())
        else:
            max_center_dist = 0.0
        cluster_max_radii.append(max_center_dist)

        mean_feat   = features.mean(axis=0)
        min_feat    = features.min(axis=0)
        max_feat    = features.max(axis=0)
        std_feat    = features.std(axis=0)
        median_feat = np.median(features, axis=0)
        count_feat  = np.array([len(node_indices)])
        count_all.append(count_feat)

        full_feature = np.concatenate([count_feat, mean_feat, min_feat, max_feat, std_feat, median_feat])
        supernode_feature_list.append(full_feature)
        supernode_centroid_list.append(centroid)
        supernode_to_nuclei_ids[count] = N_used[node_indices].tolist()
        count += 1

    if cluster_max_radii:
        radii = np.array(cluster_max_radii)
        print(f"[radii] mean={radii.mean():.2f} px  median={np.median(radii):.2f} px  "
              f"min={radii.min():.2f} px  max={radii.max():.2f} px")

    supernode_features  = np.array(supernode_feature_list) if len(supernode_feature_list) else np.zeros((0, 1 + 5 * X_data.shape[1]))
    supernode_centroids = np.array(supernode_centroid_list) if len(supernode_centroid_list) else np.zeros((0, centroids.shape[1]))
    count_all           = np.array(count_all) if len(count_all) else np.zeros((0,1))

    print("supernode formation:", len(valid_labels))
    if count_all.size > 0:
        print("count_all mean", np.mean(count_all, axis=0), np.median(count_all, axis=0), np.min(count_all, axis=0), np.max(count_all, axis=0))
    else:
        print("count_all mean", "n/a", "n/a")

    return supernode_labels, supernode_features, supernode_centroids, supernode_to_nuclei_ids
