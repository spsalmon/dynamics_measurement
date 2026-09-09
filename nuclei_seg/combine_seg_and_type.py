import os
from towbintools.foundation.file_handling import get_dir_filemap, add_dir_to_experiment_filemap
from towbintools.foundation.image_handling import read_tiff_file
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

mask_dir = "/mnt/towbin.data/shared/spsalmon/20260807_134551_682_ZIVA_60x_col10_reporter/analysis_stacks/ch2_seg_cellpose"
nuclei_type_dir = "/mnt/towbin.data/shared/spsalmon/20260807_134551_682_ZIVA_60x_col10_reporter/analysis_stacks/ch1_ch2_seg"

mask_column_name = os.path.basename(mask_dir)
nuclei_type_column_name = os.path.basename(nuclei_type_dir)
output_dir = "/mnt/towbin.data/shared/spsalmon/20260807_134551_682_ZIVA_60x_col10_reporter/analysis_stacks/ch2_nuclei_type"

os.makedirs(output_dir, exist_ok=True)

filemap = get_dir_filemap(mask_dir)
filemap = filemap.rename({"ImagePath": mask_column_name})
filemap = add_dir_to_experiment_filemap(filemap, nuclei_type_dir, nuclei_type_column_name)

# filter out columns containing missing values
cols = [mask_column_name, nuclei_type_column_name]
filemap = filemap.drop_nulls(subset=cols)

type_id_to_type = {
    0: "background",
    1: "epidermis",
    2: "intestine",
    3: "other",
}

def process_row(row):
    mask_path = row[mask_column_name]
    nuclei_type_path = row[nuclei_type_column_name]

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
        label_mask = (mask == label) & nuclei_type_mask
        sum_for_each_type = {type_id: np.sum(nuclei_type[label_mask] == type_id) for type_id in type_id_to_type.keys()}
        type_of_label = max(sum_for_each_type, key=sum_for_each_type.get) if sum(sum_for_each_type.values()) > 0 else np.nan
        if np.isnan(type_of_label):
            label_to_type[label] = "background"
        else:
            label_to_type[label] = type_id_to_type.get(np.round(type_of_label).astype(int), "unknown")
    
    label_df = pd.DataFrame(list(label_to_type.items()), columns=["Label", "Type"])
    label_df.to_csv(output_path, index=False)

Parallel(n_jobs=8)(delayed(process_row)(row) for row in tqdm(filemap.iter_rows(named=True), desc="Combining masks and types", total=len(filemap)))


