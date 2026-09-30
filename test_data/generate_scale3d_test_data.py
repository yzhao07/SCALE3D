#!/usr/bin/env python3
"""Create small, aligned SCALE3D fixtures from the three original specimens.

The script deliberately refuses to overwrite any output.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


OUTPUT_ROOT = Path(__file__).resolve().parent
CROP_SHAPE_ZXY = (64, 256, 256)

SAMPLES = (
    {
        "id": "AFM076",
        "name": "OTLS4_NODO_5-11-23_AFM076_well_8",
        "label": 1,
        "h5": Path("/raid60/projects/UPenn_OTLS4/OTLS4_NODO_5-11-23_AFM076_well_8/fused_corrected1.h5"),
        "gland": Path("/raid60/projects/UPenn_OTLS4/OTLS4_NODO_5-11-23_AFM076_well_8/nnunet/results913/stitched/afm076_stitched_glands.nii.gz"),
        "origin_zxy": (448, 256, 0),
    },
    {
        "id": "AFM081",
        "name": "OTLS4_NODO_5-15-23_AFM081_well_5",
        "label": 0,
        "h5": Path("/raid60/projects/UPenn_OTLS4/OTLS4_NODO_5-15-23_AFM081_well_5/fused_corrected1.h5"),
        "gland": Path("/raid60/projects/UPenn_OTLS4/OTLS4_NODO_5-15-23_AFM081_well_5/nnunet/results913/stitched/stitched_volume.nii.gz"),
        "origin_zxy": (64, 2304, 2304),
    },
    {
        "id": "EAM009",
        "name": "OTLS4_NODO_9-13-23_EAM009_well_5",
        "label": 1,
        "h5": Path("/raid60/projects/UPenn_OTLS4/OTLS4_NODO_9-13-23_EAM009_well_5/fused_corrected1.h5"),
        "gland": Path("/raid60/projects/UPenn_OTLS4/OTLS4_NODO_9-13-23_EAM009_well_5/nnunet/results913/stitched/stitched_volume.nii.gz"),
        "origin_zxy": (192, 1792, 2304),
    },
)


def ensure_new(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing output(s): " + ", ".join(existing))


def crop_sample(sample: dict[str, object]) -> dict[str, object]:
    sample_dir = OUTPUT_ROOT / str(sample["name"])
    output_h5 = sample_dir / "image.h5"
    output_gland = sample_dir / "gland_mask.nii.gz"
    ensure_new([sample_dir, output_h5, output_gland])
    sample_dir.mkdir(parents=False)

    z0, x0, y0 = sample["origin_zxy"]
    dz, dx, dy = CROP_SHAPE_ZXY
    source_h5 = Path(sample["h5"])
    with h5py.File(source_h5, "r") as source, h5py.File(output_h5, "w") as target:
        for channel in ("s00", "s01"):
            source_data = source[f"t00000/{channel}/0/cells"]
            crop = source_data[z0 : z0 + dz, x0 : x0 + dx, y0 : y0 + dy]
            if crop.shape != CROP_SHAPE_ZXY:
                raise ValueError(f"Unexpected {sample['id']} {channel} crop shape: {crop.shape}")
            dataset = target.create_dataset(
                f"t00000/{channel}/0/cells",
                data=crop,
                compression="gzip",
                compression_opts=6,
                shuffle=True,
                chunks=(8, 128, 128),
            )
            dataset.attrs["source_path"] = str(source_h5)
            dataset.attrs["source_origin_zxy"] = (z0, x0, y0)

    source_gland = nib.load(sample["gland"])
    # Source gland masks are (x, y, z); SCALE3D transposes them to (z, x, y).
    gland_crop = np.asarray(
        source_gland.dataobj[x0 : x0 + dx, y0 : y0 + dy, z0 : z0 + dz],
        dtype=source_gland.get_data_dtype(),
    )
    if gland_crop.shape != (dx, dy, dz):
        raise ValueError(f"Unexpected {sample['id']} gland crop shape: {gland_crop.shape}")
    affine = source_gland.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(source_gland.affine, (x0, y0, z0))
    nib.save(nib.Nifti1Image(gland_crop, affine, source_gland.header), output_gland)

    return {
        "name": sample["name"],
        "id": sample["id"],
        "BCR5yr": sample["label"],
        "OriginalPath": output_h5.relative_to(OUTPUT_ROOT.parent).as_posix(),
        "StartIdx": 0,
        "endIdx": dz,
        "GlandMaskPath": output_gland.relative_to(OUTPUT_ROOT.parent).as_posix(),
        "source_h5": str(source_h5),
        "source_gland_mask": str(sample["gland"]),
        "source_z": z0,
        "source_x": x0,
        "source_y": y0,
        "depth": dz,
        "height": dx,
        "width": dy,
    }


def main() -> None:
    output_csv = OUTPUT_ROOT / "scale3d_test_samples.csv"
    ensure_new([output_csv] + [OUTPUT_ROOT / str(sample["name"]) for sample in SAMPLES])
    rows = [crop_sample(sample) for sample in SAMPLES]
    with output_csv.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(rows)} SCALE3D fixtures under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
