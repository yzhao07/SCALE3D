import argparse
import os
import numpy as np
import anndata as ad
import pandas as pd
import squidpy as sq
import scanpy as sc
from joblib import Parallel, delayed

def fast_extract_biological_interactions(X, centroids, cell_types, radius=120):
    """
    Compute Squidpy graph-level features.
    """
    adata = ad.AnnData(X=X, dtype=np.float32)
    adata.obs["cell_type"] = pd.Categorical(cell_types)
    adata.obsm["spatial"] = centroids.astype(np.float32)

    sq.gr.spatial_neighbors(adata, coord_type="generic", radius=radius)
    sq.gr.interaction_matrix(adata, cluster_key="cell_type", copy=False)
    sq.gr.nhood_enrichment(adata, cluster_key="cell_type", copy=False)
    sq.gr.centrality_scores(adata, cluster_key="cell_type", copy=False)

    cats = adata.obs["cell_type"].cat.categories
    interaction = adata.uns["cell_type_interactions"]
    if not isinstance(interaction, pd.DataFrame):
        interaction = pd.DataFrame(interaction, index=cats, columns=cats)

    enrichment = adata.uns["cell_type_nhood_enrichment"]["zscore"]
    if not isinstance(enrichment, pd.DataFrame):
        enrichment = pd.DataFrame(enrichment, index=cats, columns=cats)

    centrality = adata.uns["cell_type_centrality_scores"]
    if not isinstance(centrality, pd.DataFrame):
        centrality = pd.DataFrame(centrality, index=cats, columns=centrality.columns)

    A = adata.obsp["spatial_connectivities"]
    degrees = np.array(A.sum(axis=1)).flatten()
    node_count = A.shape[0]
    if node_count <= 1:
        density = 0.0
    else:
        density = A.nnz / (node_count * (node_count - 1))
    avg_degree = degrees.mean() if node_count > 0 else 0.0
    clustering_coeff = (A @ A @ A).diagonal().sum() / (degrees.sum() + 1e-6)

    features = {
        "graph_density": density,
        "graph_avg_degree": avg_degree,
        "graph_clustering_coeff": clustering_coeff,
        "node_count": node_count,
    }

    for c1 in interaction.index:
        for c2 in interaction.columns:
            features[f"inter_freq_{c1}_{c2}"] = interaction.loc[c1, c2]

    for c1 in enrichment.index:
        for c2 in enrichment.columns:
            features[f"enrichZ_{c1}_{c2}"] = enrichment.loc[c1, c2]

    for ctype in centrality.index:
        for metric in centrality.columns:
            features[f"centrality_{ctype}_{metric}"] = centrality.loc[ctype, metric]

    return pd.Series(features, dtype=np.float32)


def process_sample(
    sample_name,
    metadata,
    epi_adata,
    stromal_adata,
    n_epi,
    feature_file_prefix,
    supercell_root,
    radius,
    block_size,
    label_column,
):

    try:
        label_values = metadata.loc[metadata["name"] == sample_name, label_column].values
        if len(label_values) == 0:
            raise ValueError(f"sample {sample_name} not found in metadata")
        label = int(label_values[0])

        adata_subset_epi = epi_adata[epi_adata.obs["slide_id"] == sample_name].copy()
        adata_subset_stromal = stromal_adata[stromal_adata.obs["slide_id"] == sample_name].copy()

        supercell_dir = os.path.join(supercell_root, sample_name, "supercell")
        if not os.path.isdir(supercell_dir):
            return None

        file_list = sorted(
            os.path.join(supercell_dir, filename)
            for filename in os.listdir(supercell_dir)
            if filename.startswith(feature_file_prefix) and filename.endswith(".npz")
        )

        all_feats = []
        for file_path in file_list:
            parts = os.path.basename(file_path).split("_")[-2:]
            blk_x, blk_y = int(parts[0]), int(parts[1].split(".")[0])
            all_x = []
            final_cluster = []
            centroids_blk = []

            X = np.load(file_path, allow_pickle=True)["arr_0"]
            if X.shape[0] == 0:
                continue
            cell_type = X[:, -1].astype(int)

            X_feat_epi = X[cell_type == 1, :]
            centroids_epi = adata_subset_epi.obsm["centroids"]
            mask_epi = (
                (centroids_epi[:, -2] > blk_x * block_size)
                & (centroids_epi[:, -2] < (blk_x + 1) * block_size)
                & (centroids_epi[:, -1] > blk_y * block_size)
                & (centroids_epi[:, -1] < (blk_y + 1) * block_size)
            )
            adata_subset_blk_epi = adata_subset_epi[mask_epi]
            leiden_clusters_epi = adata_subset_blk_epi.obs["leiden_harmony"].astype(int).to_numpy()
            print(f"epi size {np.unique(leiden_clusters_epi)}")
            assert len(leiden_clusters_epi) == X_feat_epi.shape[0], (
                f"Epithelial Mismatch in block {blk_x}_{blk_y}"
            )
            all_x.append(X_feat_epi)
            final_cluster.extend(leiden_clusters_epi)
            centroids_blk.append(centroids_epi[mask_epi])

            X_feat_stromal = X[cell_type == 0, :]
            centroids_stromal = adata_subset_stromal.obsm["centroids"]
            mask_stromal = (
                (centroids_stromal[:, -2] > blk_x * block_size)
                & (centroids_stromal[:, -2] < (blk_x + 1) * block_size)
                & (centroids_stromal[:, -1] > blk_y * block_size)
                & (centroids_stromal[:, -1] < (blk_y + 1) * block_size)
            )
            adata_subset_blk_stromal = adata_subset_stromal[mask_stromal]
            leiden_clusters_stromal = (
                adata_subset_blk_stromal.obs["leiden_harmony"].astype(int).to_numpy() + n_epi
            )
            print(f"stromal size {np.unique(leiden_clusters_stromal)}")
            assert len(leiden_clusters_stromal) == X_feat_stromal.shape[0], (
                f"Stromal Mismatch in block {blk_x}_{blk_y}"
            )
            all_x.append(X_feat_stromal)
            final_cluster.extend(leiden_clusters_stromal)
            centroids_blk.append(centroids_stromal[mask_stromal])

            final_cluster = np.array(final_cluster)
            X = np.concatenate(all_x, axis=0)
            centroids_blk = np.concatenate(centroids_blk, axis=0)

            feat = fast_extract_biological_interactions(
                X[:, :266], centroids_blk, final_cluster, radius=radius
            )
            all_feats.append(feat)

        if len(all_feats) == 0:
            return None

        df_blk = pd.concat(all_feats, axis=1)
        stats = pd.concat(
            [
                df_blk.mean(axis=1).rename("mean"),
                df_blk.min(axis=1).rename("min"),
                df_blk.max(axis=1).rename("max"),
                df_blk.std(axis=1).rename("std"),
                df_blk.median(axis=1).rename("median"),
            ],
            axis=1,
        ).T

        flat = stats.values.flatten()
        feat_names = [f"{stat}_{name}" for stat in stats.index for name in df_blk.index]
        out = pd.Series(flat, index=feat_names, name=sample_name)
        out["sample"] = sample_name
        out["label"] = label
        return out
    except Exception as exc:
        print(f"[ERROR] {sample_name}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract graph-level supercell interaction features for a cohort."
    )
    parser.add_argument("--metadata_csv", type=str, required=True, help="metadata CSV path")
    parser.add_argument("--epi_h5ad", type=str, required=True, help="epithelial clustering h5ad")
    parser.add_argument("--stromal_h5ad", type=str, required=True, help="stromal clustering h5ad")
    parser.add_argument(
        "--feature_file_prefix",
        type=str,
        required=True,
        help="supercell feature filename prefix",
    )
    parser.add_argument(
        "--supercell_root",
        type=str,
        required=True,
        help="root directory containing <sample>/supercell folders",
    )
    parser.add_argument("--output_csv", type=str, required=True, help="output CSV path")
    parser.add_argument("--radius", type=int, default=400, help="spatial radius for graph")
    parser.add_argument("--n_jobs", type=int, default=6, help="parallel sample workers")
    parser.add_argument("--label_column", type=str, default="BCR5yr", help="label column in metadata CSV")
    parser.add_argument( "--sample_column", type=str, default="name", help="sample name column in metadata CSV")
    parser.add_argument("--block_size", type=int, default=1024, help="supercell block size")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_csv)
    if args.sample_column != "name":
        metadata = metadata.rename(columns={args.sample_column: "name"})
    if "name" not in metadata.columns:
        raise ValueError("metadata CSV must contain a sample column")
    if args.label_column not in metadata.columns:
        raise ValueError(f"metadata CSV is missing label column: {args.label_column}")

    epi_adata = sc.read_h5ad(args.epi_h5ad)
    stromal_adata = sc.read_h5ad(args.stromal_h5ad)
    n_epi = len(epi_adata.obs["leiden_harmony"].unique())

    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_sample)(
            sample_name=sample_name,
            metadata=metadata,
            epi_adata=epi_adata,
            stromal_adata=stromal_adata,
            n_epi=n_epi,
            feature_file_prefix=args.feature_file_prefix,
            supercell_root=args.supercell_root,
            radius=args.radius,
            block_size=args.block_size,
            label_column=args.label_column,
        )
        for sample_name in metadata["name"]
    )

    results = [result for result in results if result is not None]
    if not results:
        raise RuntimeError("no graph features were generated for any sample")

    df_features = pd.DataFrame(results)
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df_features.to_csv(args.output_csv, index=False)
    print("Done")


if __name__ == "__main__":
    main()
