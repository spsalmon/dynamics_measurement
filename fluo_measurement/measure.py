import numpy as np
from tqdm import tqdm
import os
from tifffile import imread
from towbintools.foundation.file_handling import get_dir_filemap, add_dir_to_experiment_filemap
import cv2
from joblib import Parallel, delayed
from time import perf_counter
import pandas as pd
from scipy import ndimage as ndi
import polars as pl

from queue import Queue
from threading import Thread

from towbintools.foundation.file_handling import read_filemap
 
 
def prefetch_stacks(raw_paths, mask_paths, prefetch=8, channel=0):
    q = Queue(maxsize=prefetch)

    def producer():
        for raw_path, mask_path in zip(raw_paths, mask_paths):
            raw = imread(raw_path)[:, channel]
            mask = imread(mask_path)
            q.put((raw_path, raw, mask))
        q.put(None)

    Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        yield item

def _dilate_labels(label_image, small_kernel, big_kernel):
    """
    For each nucleus label, creates a 'donut' mask in the region between
    small_kernel and big_kernel dilation distances from the nucleus boundary.
    This peri-nuclear ring can be used to measure cytoplasmic signal while
    avoiding the nucleus itself and pixels too close to it (reducing
    under-segmentation artifacts).

    Label identity is preserved via nearest-neighbor propagation (distance
    transform), so overlapping donuts from adjacent nuclei are resolved by
    proximity.
    """
    # Grey dilation expands label values, but conflicts at label borders need resolution.
    # Approach: for each label, we need a proper expanded mask without cross-contamination.
    # Fastest correct approach: dilate the binary occupied mask, then propagate labels
    # via nearest-neighbor (voronoi-like) only within the dilated footprint.
    occupied = label_image > 0
    small_dilated_occupied = cv2.dilate(occupied.astype(np.uint8), small_kernel).astype(bool)
    big_dilated_occupied = cv2.dilate(occupied.astype(np.uint8), big_kernel).astype(bool)
    donut_occupied = np.logical_and(big_dilated_occupied, ~small_dilated_occupied)

    # Propagate labels into the dilated region via nearest-label distance transform
    # ndi.distance_transform_edt on the inverted label mask gives nearest-foreground coords
    _, nearest_idx = ndi.distance_transform_edt(label_image == 0, return_indices=True)
    expanded_labels = label_image[tuple(nearest_idx)]  # nearest label for every pixel
    donut_labels = expanded_labels * donut_occupied  # only keep labels in the donut region, zero elsewhere

    return donut_labels
 
def process_plane(plane, mask, small_kernel, big_kernel, plane_index, camera_min=100.0):
    if not np.any(mask):
        return []

    raw = plane.astype(np.float32) - camera_min
    raw[raw < 0] = 0
 
    labels = np.unique(mask)
    labels = labels[labels > 0]

    # centroid per label
    centroids = ndi.center_of_mass(mask > 0, mask, labels)
 
    nucleus_means   = ndi.mean(raw,   mask, labels)
    nucleus_medians = ndi.median(raw, mask, labels)

    nucleus_sizes = ndi.sum(mask > 0, mask, labels)
 
    expanded = _dilate_labels(mask, small_kernel, big_kernel)
    nuclei_footprint = mask > 0
    cytoplasm_labels = expanded.copy()
    cytoplasm_labels[nuclei_footprint] = 0
 
    cyto_means   = ndi.mean(raw,   cytoplasm_labels, labels)
    cyto_medians = ndi.median(raw, cytoplasm_labels, labels)
 
    rows = []
    for i, lbl in enumerate(labels):
        cm   = float(cyto_means[i])
        cmed = float(cyto_medians[i])
        nm   = float(nucleus_means[i])
        nmed = float(nucleus_medians[i])
        size = float(nucleus_sizes[i])
        rows.append({
            "Z":                             plane_index,
            "Label":                         int(lbl),
            "Size":                          size,
            "CentroidX":                     centroids[i][1],
            "CentroidY":                     centroids[i][0],
            "MeanIntensityNucleus":          nm,
            "MedianIntensityNucleus":        nmed,
            "MeanIntensityCytoplasm":        cm,
            "MedianIntensityCytoplasm":      cmed,
            "NucleusCytoplasmRatioMean":     nm / cm   if cm   else np.nan,
            "NucleusCytoplasmRatioMedian":   nmed / cmed if cmed else np.nan,
        })
 
    return rows
 
 
def measure_stack_nuclear_stats(
    raw_stack,
    mask_stack,
    small_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    big_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    n_jobs=1,
):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_plane)(plane, mask, small_kernel, big_kernel, i)
        for i, (plane, mask) in enumerate(zip(raw_stack, mask_stack))
    )
    all_stats = [item for sublist in results for item in sublist]
    return pd.DataFrame(all_stats)

experiment_dir = "/mnt/towbin.data/shared/spsalmon/20251023_115945_091_ZIVA_60x_397_405_yap_dynamics"
raw_dir = os.path.join(experiment_dir, "raw_stacks")
raw_dir_name = os.path.basename(raw_dir)
analysis_dir = os.path.join(experiment_dir, "analysis_stacks")
report_dir = os.path.join(experiment_dir, "analysis", "report")
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

channel = 0
rerun = True

experiment_filemap = get_dir_filemap(raw_dir)
experiment_filemap = experiment_filemap.rename({"ImagePath": raw_dir_name})

mask_dir = os.path.join(analysis_dir, "ch2_seg_cellpose_stitched")
experiment_filemap = add_dir_to_experiment_filemap(experiment_filemap, mask_dir, os.path.basename(mask_dir))

classification_dir = None
output_dir = os.path.join(analysis_dir, "ch1_cellpose_stitched_measurements")
os.makedirs(output_dir, exist_ok=True)

analysis_filemap_path = [os.path.join(report_dir, f) for f in os.listdir(report_dir) if "analysis_filemap_annotated" in f]
print(f'Number of files in experiment filemap: {len(experiment_filemap)}')
if len(analysis_filemap_path) > 0:
    analysis_filemap = read_filemap(analysis_filemap_path[0]).select(["Point", "Time", "Ignore"])
    experiment_filemap = experiment_filemap.join(analysis_filemap, on=["Point", "Time"], how="left").filter(~pl.col("Ignore")).drop("Ignore")

print(f'Number of files to process after filtering: {len(experiment_filemap)}')

# Filter out rows where mask is missing or output already exists
rows_to_keep = []
for row in experiment_filemap.iter_rows(named=True):
    mask_col = os.path.basename(mask_dir)
    if row[mask_col] is None or row[mask_col] == "":
        continue
    output_file_path = os.path.join(output_dir, os.path.basename(row[raw_dir_name]).replace(".ome.tiff", ".csv"))
    if not os.path.exists(output_file_path) or rerun:
        rows_to_keep.append(row)

experiment_filemap = pl.DataFrame(rows_to_keep)

for raw_path, raw_stack, mask_stack in tqdm(prefetch_stacks(experiment_filemap[raw_dir_name], experiment_filemap[os.path.basename(mask_dir)], channel=channel), total=len(experiment_filemap[raw_dir_name])):

    output_file_path = os.path.join(output_dir, os.path.basename(raw_path).replace(".ome.tiff", ".csv"))
    stats_df = measure_stack_nuclear_stats(raw_stack, mask_stack, n_jobs=8)
    stats_df.to_csv(output_file_path, index=False)