import os
from towbintools.foundation.file_handling import get_dir_filemap, add_dir_to_experiment_filemap
from towbintools.foundation.image_handling import read_tiff_file
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import polars as pl

measurements_dir = "/mnt/towbin.data/shared/spsalmon/20251023_115945_091_ZIVA_60x_397_405_yap_dynamics/analysis_stacks/ch1_cellpose_stitched_measurements"
nuclei_type_dir = "/mnt/towbin.data/shared/spsalmon/20251023_115945_091_ZIVA_60x_397_405_yap_dynamics/analysis_stacks/ch2_nuclei_type"

output_dir = "/mnt/towbin.data/shared/spsalmon/20251023_115945_091_ZIVA_60x_397_405_yap_dynamics/analysis_stacks/ch1_cellpose_final_measurements"
os.makedirs(output_dir, exist_ok=True)

filemap = get_dir_filemap(measurements_dir)
filemap = filemap.rename({"ImagePath": "ch1_cellpose_stitched_measurements"})
filemap = add_dir_to_experiment_filemap(filemap, nuclei_type_dir, "ch2_nuclei_type")
cols = ["ch2_nuclei_type", "ch1_cellpose_stitched_measurements"]
filemap = filemap.drop_nulls(subset=cols)

def process_row(row):
    try:
        measurements_path = row["ch1_cellpose_stitched_measurements"]
        nuclei_type_path = row["ch2_nuclei_type"]

        output_path = os.path.join(output_dir, os.path.basename(measurements_path))

        if os.path.exists(output_path):
            print(f"Output already exists for {measurements_path}, skipping.")
            return

        measurements_df = pl.read_csv(measurements_path)
        nuclei_type = pl.read_csv(nuclei_type_path)

        measurements_with_type = measurements_df.join(nuclei_type, left_on="Label", right_on="Label", how="left")

        # for each unique label, keep only the row with the maximum size
        measurements_with_type = measurements_with_type.sort("Size", descending = True).unique(subset="Label", keep="first").sort("Label")

        measurements_with_type.write_csv(output_path)
    except Exception as e:
        print(f"Error processing row {row}: {e}")

Parallel(n_jobs=-1)(delayed(process_row)(row) for row in tqdm(filemap.iter_rows(named=True), total=len(filemap), desc="Refining measurements"))