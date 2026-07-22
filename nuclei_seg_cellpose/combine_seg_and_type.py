import os
from towbintools.foundation.file_handling import get_dir_filemap, add_dir_to_experiment_filemap
from towbintools.foundation.image_handling import read_tiff_file
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import polars as pl

mask_dir = "/mnt/towbin.data/shared/spsalmon/20251023_115945_091_ZIVA_60x_397_405_yap_dynamics/analysis_stacks/ch2_seg_cellpose_stitched"
nuclei_type_dir = "/mnt/towbin.data/shared/spsalmon/20251023_115945_091_ZIVA_60x_397_405_yap_dynamics/analysis_stacks/ch1_ch2_seg"
measurements_dir = "/mnt/towbin.data/shared/spsalmon/20251023_115945_091_ZIVA_60x_397_405_yap_dynamics/analysis_stacks/ch1_cellpose_stitched_measurements"

stitch_3D = True
output_stitched_dir = "/mnt/towbin.data/shared/spsalmon/20251023_115945_091_ZIVA_60x_397_405_yap_dynamics/analysis_stacks/ch2_seg_cellpose_stitched"

output_dir = "/mnt/towbin.data/shared/spsalmon/20251023_115945_091_ZIVA_60x_397_405_yap_dynamics/analysis_stacks/ch2_nuclei_type"

if stitch_3D:
    os.makedirs(output_stitched_dir, exist_ok=True)

os.makedirs(output_dir, exist_ok=True)

filemap = get_dir_filemap(mask_dir)
filemap = filemap.rename({"ImagePath": "ch2_seg_cellpose"})
filemap = add_dir_to_experiment_filemap(filemap, nuclei_type_dir, "ch1_ch2_seg")
filemap = add_dir_to_experiment_filemap(filemap, measurements_dir, "ch1_cellpose_stitched_measurements")

# filter out columns containing missing values
cols = ["ch2_seg_cellpose", "ch1_ch2_seg", "ch1_cellpose_stitched_measurements"]
filemap = filemap.drop_nulls(subset=cols)

type_id_to_type = {
    1: "epidermis",
    2: "intestine",
    3: "other",
}
print(filemap)

# def stitch_mask(mask_path):
#     stitched_mask_path = os.path.join(output_stitched_dir, os.path.basename(mask_path))
#     if not os.path.exists(stitched_mask_path):
#         mask = read_tiff_file(mask_path)
#         stitched_mask = stitch3D(mask)
#         imwrite(stitched_mask_path, stitched_mask, compression="zlib")

# Parallel(n_jobs=-1)(delayed(stitch_mask)(mask_path) for mask_path in tqdm(filemap["ch2_seg_cellpose"], desc="Stitching masks", total=len(filemap)))

def process_row(row):
    mask_path = row["ch2_seg_cellpose"]
    nuclei_type_path = row["ch1_ch2_seg"]

    output_path = os.path.join(output_dir, os.path.basename(mask_path).replace(".ome.tiff", ".csv"))
    if os.path.exists(output_path):
        print(f"Output already exists for {mask_path}, skipping.")
        return
    
    mask = read_tiff_file(mask_path)
    nuclei_type = read_tiff_file(nuclei_type_path)

    nuclei_type = nuclei_type * (mask > 0)
    nuclei_type_mask = (nuclei_type > 0)

    label_to_type = {}

    for label in np.unique(mask):
        if label == 0:
            continue
        type_of_label = np.mean(nuclei_type[(mask == label) & nuclei_type_mask])
        if np.isnan(type_of_label):
            label_to_type[label] = "background"
        else:
            label_to_type[label] = type_id_to_type.get(np.round(type_of_label).astype(int), "unknown")
    
    label_df = pd.DataFrame(list(label_to_type.items()), columns=["Label", "Type"])
    label_df.to_csv(output_path, index=False)

Parallel(n_jobs=8)(delayed(process_row)(row) for row in tqdm(filemap.iter_rows(named=True), desc="Combining masks and types", total=len(filemap)))


