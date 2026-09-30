# SCALE3D

Scalable 3D pathology cell-interaction analysis via supercell graphs for prostate cancer risk stratification.

![SCALE3D overview](img.png)

## Overview

SCALE3D processes multiplexed 3D pathology images in four stages:

1. **Nuclei segmentation and feature extraction** segments nuclei and calculates per-nucleus morphology, intensity, texture, neighborhood, gland, and spatial features.
2. **Supercell formation** groups spatially adjacent and phenotypically similar nuclei into epithelial or stromal supercells.
3. **Supercell subtyping** learns cohort-level epithelial and stromal supercell subtypes using PCA, Harmony, a neighbor graph, and Leiden clustering.
4. **Graph construction and feature extraction** constructs spatial supercell graphs and aggregates their interaction, enrichment, centrality, and topology measurements into one feature vector per specimen.

| Stage | Main input | Main output |
| --- | --- | --- |
| 1. Segmentation and nuclear features | 3D HDF5 volume and gland mask | Nucleus masks and per-nucleus arrays |
| 2. Supercell formation | Per-nucleus arrays and fitted scaler | Per-supercell arrays and membership maps |
| 3. Supercell subtyping | Cohort supercell arrays | Epithelial and stromal `.h5ad` files |
| 4. Graph features | Supercells, subtype files, and metadata | Specimen-level feature CSV |

## Repository structure

```text
SCALE3D/
├── 01.Nulcei_segmentation_and_feature_extraction/
│   ├── Nuclei_segmentation.py
│   ├── Feature_extraction.py
│   ├── model/
│   └── utils/
├── 02.Supercell_formation/
│   ├── super_cell_identification.py
│   └── supercell_graph/
├── 03.Supercell_subtyping/cluster_supercell.py
├── 04.Graph_construction_and_graph_feature_extraction/
│   └── graph_formation_and_graph_feature_extraction.py
└── Evaluation/
```

## Requirements

The repository does not yet include a pinned environment file. The scripts require these major packages:

- Python 3, NumPy, pandas, SciPy, scikit-image, scikit-learn, and joblib
- h5py and nibabel
- PyTorch and Cellpose with a CUDA-capable GPU for 3D segmentation
- Scanpy and AnnData
- CuPy, RAPIDS SingleCell, and RMM with compatible CUDA versions for subtyping
- Squidpy for spatial graph construction

Stage 3 currently requires a CUDA GPU and has no CPU fallback. Install mutually compatible CUDA, PyTorch, Cellpose, CuPy, RAPIDS SingleCell, and RMM versions for the target system. Run the commands below from the repository root.

## Recommended data layout

```text
DATA_ROOT/
├── sample_A/
│   ├── image.h5
│   ├── gland_mask.nii.gz
│   ├── cellpose/
│   ├── individual_cells/
│   └── supercell/
├── sample_B/
│   └── ...
├── samples.csv
├── metadata.csv
├── normalization/feat53_mean_std_scaler.pkl
├── subtypes/
└── features/
```

## 1. Nuclei segmentation and nuclear feature extraction

### 1.1 Nuclei segmentation

Script: `01.Nulcei_segmentation_and_feature_extraction/Nuclei_segmentation.py`

#### Input

The multiplexed 3D HDF5 input must use this hierarchy:
```text
/t00000/<channel>/<resolution>/cells
```

#### Processing and output

Each block is rescaled and filtered, then processed by the Cellpose `nuclei` model with `denoise_nuclei` restoration in 3D. The output is:

```text
cellpose/
├── mask_blk_<x>_<y>.nii.gz
├── imgdn_blk_<x>_<y>.nii.gz
├── mask_blk_<x>_<y>.avi       # optional
└── imgdn_blk_<x>_<y>.avi      # optional
```

#### Example

```bash
python 01.Nulcei_segmentation_and_feature_extraction/Nuclei_segmentation.py \
  --h5 /path/to/DATA_ROOT/sample_A/image.h5 \
  --outdir /path/to/DATA_ROOT/sample_A/cellpose
```

### 1.2 Nuclear feature extraction

Script: `01.Nulcei_segmentation_and_feature_extraction/Feature_extraction.py`

#### Input

This step requires the original HDF5 file, the `cellpose/` directory, an aligned 3D gland-mask NIfTI, and a cytoplasmic channel. 

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--h5` | Yes | — | Original HDF5 volume |
| `--CellposeDir` | Yes | — | Segmentation directory |
| `--gland` | Yes | — | Aligned gland-mask NIfTI |

The code clears border-touching nuclei, removes objects smaller than 150 voxels, and calculates morphology, nuclear/cytoplasmic intensity, crowdedness, entropy, GLCM texture, gland membership, and coordinates.

#### Output

```text
individual_cells/
├── feature53_blk_<x>_<y>.npz
└── label53_blk_<x>_<y>.npz
```

The feature53_blk array contain 12 morphology, 10 intensity/crowdedness, 1 entropy, and 30 three-plane GLCM features. The label array contains the connected-component ID corresponding to every row.

#### Example

```bash
python 01.Nulcei_segmentation_and_feature_extraction/Feature_extraction.py \
  --h5 /path/to/DATA_ROOT/sample_A/image.h5 \
  --CellposeDir /path/to/DATA_ROOT/sample_A/cellpose \
  --gland /path/to/DATA_ROOT/sample_A/gland_mask.nii.gz

```

## 2. Supercell formation

Script: `02.Supercell_formation/super_cell_identification.py`

### Input

This stage consumes `feature53_blk_<x>_<y>.npz` and `label53_blk_<x>_<y>.npz`.

### Output

Default-style names are:

```text
supercell/
├── supercell_1024_feature53_d30_blk_0_0.npz
└── supercell_1024_nucleiId53_d30_blk_0_0.npy
```
Each feature array has **270 columns**:

| Columns | Contents |
| --- | --- |
| `0` | Nucleus count within each supercell |
| `1:54` | Mean of 53 nucleus features |
| `54:107` | Minimum |
| `107:160` | Maximum |
| `160:213` | Standard deviation |
| `213:266` | Median |
| `266:269` | Local `(z, x, y)` centroid |
| `269` | `1` epithelial or `0` stromal |

The membership `.npy` is a pickled dictionary from supercell row index to nucleus IDs:

```python
membership = np.load("nucleiId_file.npy", allow_pickle=True).item()
```

### Example

```bash
python 02.Supercell_formation/super_cell_identification.py \
  --cellseg /path/to/DATA_ROOT/sample_A/cellpose \
  --individual_cells_dir /path/to/DATA_ROOT/sample_A/individual_cells \
  --out_folder /path/to/DATA_ROOT/sample_A/supercell
```

## 3. Supercell subtyping

Script: `03.Supercell_subtyping/cluster_supercell.py`

This is a cohort-level step and must be run separately for epithelial and stromal supercells.

### Input

Create a sample CSV:

```csv
name
sample_A
sample_B
sample_C
```

The script converts centroids to specimen-wide coordinates and uses only columns `1:54`—the 53 mean nucleus features. It scales them, computes 20 principal components, applies Harmony using `slide_id`, and runs a neighbor graph, UMAP, and Leiden clustering on the GPU.

### Output

| AnnData location | Contents |
| --- | --- |
| `adata.X` | 53 mean features |
| `adata.obs["cell_type"]` | Compartment label |
| `adata.obs["slide_id"]` | Specimen identifier |
| `adata.obs["leiden_harmony"]` | Learned subtype |
| `adata.obsm["centroids"]` | Global `(z, x, y)` centroids |
| `adata.obsm["X_pca_harmony"]` | Harmony representation |
| `adata.obsm["X_umap_harmony"]` | UMAP coordinates |

### Epithelial example

```bash
python 03.Supercell_subtyping/cluster_supercell.py \
  --file_list /path/to/DATA_ROOT/samples.csv \
  --supercell_root /path/to/DATA_ROOT \
  --data supercell_1024_feature53_d30_blk \
  --epithelial_or_stromal 1 \
  --outfile /path/to/DATA_ROOT/subtypes/epithelial.h5ad \
  --resolution 0.1 --neighbors 50 \
  --block_size 1024 --grid_size 4 \
  --gpu_id 0 --n_jobs 16
```

### Stromal example

```bash
python 03.Supercell_subtyping/cluster_supercell.py \
  --file_list /path/to/DATA_ROOT/samples.csv \
  --supercell_root /path/to/DATA_ROOT \
  --data supercell_1024_feature53_d30_blk \
  --epithelial_or_stromal 0 \
  --outfile /path/to/DATA_ROOT/subtypes/stromal.h5ad \
  --resolution 0.1 --neighbors 50 \
  --block_size 1024 --grid_size 4 \
  --gpu_id 0 --n_jobs 16
```

## 4. Graph construction and graph feature extraction

Script: `04.Graph_construction_and_graph_feature_extraction/graph_formation_and_graph_feature_extraction.py`

### Input

This cohort-level stage requires the epithelial and stromal `.h5ad` files, stage-2 supercell arrays, and a metadata CSV.

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--metadata_csv` | Yes | — | Metadata and labels |
| `--epi_h5ad` | Yes | — | Epithelial subtypes |
| `--stromal_h5ad` | Yes | — | Stromal subtypes |
| `--feature_file_prefix` | Yes | — | Stage-2 filename prefix |
| `--supercell_root` | Yes | — | Root containing sample directories |
| `--output_csv` | Yes | — | Output feature table |
| `--radius` | Yes | `40` | Graph radius in voxel units |

### Output

The CSV has one row per successfully processed specimen. Columns follow `<aggregation>_<graph-feature>`, for example:

```text
mean_graph_density
max_graph_avg_degree
median_inter_freq_0_1
mean_enrichZ_2_4
std_centrality_3_degree_centrality
sample
label
```

This is the main SCALE3D specimen representation and can be used by the notebooks under `Evaluation/`.

### Example

```bash
python 04.Graph_construction_and_graph_feature_extraction/graph_formation_and_graph_feature_extraction.py \
  --metadata_csv /path/to/DATA_ROOT/metadata.csv \
  --epi_h5ad /path/to/DATA_ROOT/subtypes/epithelial.h5ad \
  --stromal_h5ad /path/to/DATA_ROOT/subtypes/stromal.h5ad \
  --supercell_root /path/to/DATA_ROOT \
  --feature_file_prefix supercell_1024_feature53_d30_blk \
  --output_csv /path/to/DATA_ROOT/features/scale3d_graph_features.csv \
  --label_column BCR5yr --sample_column name \
  --radius 40 --block_size 1024 --n_jobs 6
```

## Evaluation

