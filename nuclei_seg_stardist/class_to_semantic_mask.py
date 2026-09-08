import os
import numpy as np
import pandas as pd
from towbintools.foundation.image_handling import read_tiff_file
from tifffile import imwrite
from joblib import Parallel, delayed
def class_to_semantic_mask(class_df_path, mask_path, output_dir):
    class_df = pd.read_csv(class_df_path)
    label_to_class = dict(zip(class_df['Label'], class_df['Class']))

    mask = read_tiff_file(mask_path)
    semantic_mask = np.zeros_like(mask, dtype=np.uint8)

    for label, class_name in label_to_class.items():
        semantic_mask[mask == label] = class_name

    output_path = os.path.join(output_dir, os.path.basename(mask_path))
    imwrite(output_path, semantic_mask.astype(np.uint8), compression="zlib")

if __name__ == "__main__":
    class_dir = "/mnt/towbin.data/shared/spsalmon/stardist_database/60x_emr1_col10GFP_classification_dataset/classes"
    mask_dir = "/mnt/towbin.data/shared/spsalmon/stardist_database/60x_emr1_col10GFP_classification_dataset/mask_planes"
    output_dir = "/mnt/towbin.data/shared/spsalmon/stardist_database/60x_emr1_col10GFP_classification_dataset/semantic_masks"
    os.makedirs(output_dir, exist_ok=True)

    class_dfs = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
    mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
    Parallel(n_jobs=-1)(delayed(class_to_semantic_mask)(class_df, mask_file, output_dir) for class_df, mask_file in zip(class_dfs, mask_files))

