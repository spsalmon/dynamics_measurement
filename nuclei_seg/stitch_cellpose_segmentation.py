import os
from towbintools.foundation.file_handling import get_dir_filemap, add_dir_to_experiment_filemap
from towbintools.foundation.image_handling import read_tiff_file
from cellpose.utils import stitch3D
from tifffile import imwrite
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

mask_dir = "/mnt/towbin.data/shared/spsalmon/20260807_134551_682_ZIVA_60x_col10_reporter/analysis_stacks/ch2_seg_cellpose"
output_mask_dir = "/mnt/towbin.data/shared/spsalmon/20260807_134551_682_ZIVA_60x_col10_reporter/analysis_stacks/ch2_seg_cellpose_stitched"
output_mask_column_name = os.path.basename(output_mask_dir)
os.makedirs(output_mask_dir, exist_ok=True)

def stitch_mask(mask_path, output_path):
    mask = read_tiff_file(mask_path)
    stitched_mask = stitch3D(mask)
    imwrite(output_path, stitched_mask, compression="zlib")

Parallel(n_jobs=8)(
    delayed(stitch_mask)(
        os.path.join(mask_dir, filename),
        os.path.join(output_mask_dir, filename)
    )
    for filename in tqdm(os.listdir(mask_dir), desc="Stitching masks")
)