import numpy as np
from towbintools.foundation.image_handling import read_tiff_file
import os
import matplotlib.pyplot as plt
import cv2
from skimage.util import img_as_ubyte
import pandas as pd

from skimage import (
    data, restoration, util
)

from joblib import Parallel, delayed
import warnings
from towbintools.foundation.file_handling import get_dir_filemap, add_dir_to_experiment_filemap
warnings.filterwarnings('ignore')
from time import perf_counter


def process_plane(plane, mask, kernel, plane_index):
    raw = plane - 100.0
    nuclei_mask = (mask > 0).astype(int)
    nuclei_mask_bool = nuclei_mask == 1
    nuclei_labels = mask
    unique_labels = np.unique(nuclei_labels)
    unique_labels = unique_labels[unique_labels > 0]
    
    plane_stats = []

    for lbl in unique_labels:
        mask_of_label = (nuclei_labels == lbl).astype(int)
        expanded_label = (
            cv2.morphologyEx(img_as_ubyte(mask_of_label), cv2.MORPH_DILATE, kernel) > 0
        ).astype(int)
        cytoplasm_mask = (expanded_label - mask_of_label - nuclei_mask > 0).astype(int)
        cytoplasm_mask = cytoplasm_mask == 1
        
        raw_cytoplasm = raw[cytoplasm_mask]
        raw_nucleus = raw[nuclei_mask_bool]

        mean_intensity_cytoplasm = np.mean(raw_cytoplasm)
        median_intensity_cytoplasm = np.median(raw_cytoplasm)
        mean_intensity_nucleus = np.mean(raw_nucleus)
        median_intensity_nucleus = np.median(raw_nucleus)
        intensity_ratio_mean = mean_intensity_nucleus / mean_intensity_cytoplasm
        intensity_ratio_median = median_intensity_nucleus / median_intensity_cytoplasm
        
        plane_stats.append({
            "Z": plane_index,
            "Label": lbl,
            "MeanIntensityCytoplasm": mean_intensity_cytoplasm,
            "MedianIntensityCytoplasm": median_intensity_cytoplasm,
            "MeanIntensityNucleus": mean_intensity_nucleus,
            "MedianIntensityNucleus": median_intensity_nucleus,
            "NucleusCytoplasmRatioMean": intensity_ratio_mean,
            "NucleusCytoplasmRatioMedian": intensity_ratio_median,
        })
    
    all_expanded_nuclei = cv2.morphologyEx(img_as_ubyte(nuclei_mask), cv2.MORPH_DILATE, kernel)
    all_cytoplasm = (all_expanded_nuclei - nuclei_mask > 0).astype(int)
    all_cytoplasm_bool = all_cytoplasm == 1

    raw_all_nuclei = raw[nuclei_mask_bool]
    raw_all_cytoplasm = raw[all_cytoplasm_bool]

    mean_intensity_all_nuclei = np.mean(raw_all_nuclei)
    median_intensity_all_nuclei = np.median(raw_all_nuclei)
    mean_intensity_all_cytoplasm = np.mean(raw_all_cytoplasm)
    median_intensity_all_cytoplasm = np.median(raw_all_cytoplasm)
    intensity_ratio_mean_all = mean_intensity_all_nuclei / mean_intensity_all_cytoplasm
    intensity_ratio_median_all = median_intensity_all_nuclei / median_intensity_all_cytoplasm
    
    for stat in plane_stats:
        stat.update({
            'MeanIntensityAllNuclei': mean_intensity_all_nuclei,
            'MedianIntensityAllNuclei': median_intensity_all_nuclei,
            'MeanIntensityAllCytoplasm': mean_intensity_all_cytoplasm,
            'MedianIntensityAllCytoplasm': median_intensity_all_cytoplasm,
            'NucleusCytoplasmRatioMeanAll': intensity_ratio_mean_all,
            'NucleusCytoplasmRatioMedianAll': intensity_ratio_median_all
        })
    
    return plane_stats


def measure_stack_nuclear_stats(
    raw_stack,
    mask_stack,
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    n_jobs=1
):
    start = perf_counter()
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_plane)(plane, mask, kernel, i)
        for i, (plane, mask) in enumerate(zip(raw_stack, mask_stack))
    )
    
    all_stats = [item for sublist in results for item in sublist]
    print(f'Computed nuclear stats in {perf_counter() - start:.2f} s')
    return pd.DataFrame(all_stats)

img_dir = '/mnt/towbin.data/shared/spsalmon/20241115_122955_248_ZIVA_40x_raga1_full_deletion/raw/pad2/'
mask_dir = '/mnt/towbin.data/shared/spsalmon/20241115_122955_248_ZIVA_40x_raga1_full_deletion/analysis/ch2_stardist/pad2/'
classification_dir = '/mnt/towbin.data/shared/spsalmon/20241115_122955_248_ZIVA_40x_raga1_full_deletion/analysis/ch2_nuclei_types_stardist/pad2/'

filemap = get_dir_filemap(img_dir)
filemap = add_dir_to_experiment_filemap(filemap, mask_dir, "MaskPath")
filemap = add_dir_to_experiment_filemap(filemap, classification_dir, "ClassificationPath")

output_dir = '/mnt/towbin.data/shared/spsalmon/20241115_122955_248_ZIVA_40x_raga1_full_deletion/analysis/nuclear_stats_stardist/pad2/'
os.makedirs(output_dir, exist_ok=True)
# keep only rows with mask and classification
filemap = filemap[filemap['MaskPath'] != '']
filemap = filemap[filemap['ClassificationPath'] != '']

# parallelize time loop with joblib

def process_time(time, point_data):
    print(f"Processing time {time}")
    time_data = point_data[point_data['Time'] == time]
    raw_path = time_data['ImagePath'].values[0]

    output_path = os.path.join(output_dir, os.path.basename(raw_path).replace('.ome.tiff', '.csv'))
    if os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping")
        return
        
    mask_path = time_data['MaskPath'].values[0]
    classification_path = time_data['ClassificationPath'].values[0]

    raw_stack = read_tiff_file(raw_path, channels_to_keep = [1])
    mask_stack = read_tiff_file(mask_path)

    try:
        classification_df = pd.read_csv(classification_path)

        nuclei_stats_df = measure_stack_nuclear_stats(raw_stack, mask_stack, n_jobs=16)
        nuclei_stats_df = nuclei_stats_df.merge(classification_df, on=['Z', 'Label'], how='left')

        nuclei_stats_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error processing time {time}: {e}")

for point in filemap['Point'].unique():
    print(f"Processing point {point}")
    point_data = filemap[filemap['Point'] == point]

    Parallel(n_jobs=1)(
        delayed(process_time)(time, point_data) for time in point_data['Time'].unique()
    )
    # for time in point_data['Time'].unique():
    #     process_time(time, point_data)