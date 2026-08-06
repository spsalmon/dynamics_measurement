"""
Fast per-plane nucleus / peri-nuclear ("donut") intensity measurements.
"""

import os
import numpy as np
import cv2
import polars as pl
from tifffile import imread, TiffFile
from joblib import Parallel, delayed
from tqdm import tqdm
from towbintools.foundation.file_handling import get_dir_filemap, add_dir_to_experiment_filemap, read_filemap

SMALL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BIG_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

COLUMNS = [
    "Z", "Label", "Size", "CentroidX", "CentroidY",
    "MeanIntensityNucleus", "MedianIntensityNucleus",
    "MeanIntensityCytoplasm", "MedianIntensityCytoplasm",
    "NucleusCytoplasmRatioMean", "NucleusCytoplasmRatioMedian",
]


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _kernel_offsets(kernel):
    """Kernel footprint as (dy, dx) offsets sorted by euclidean distance."""
    ky, kx = np.nonzero(kernel)
    cy = (kernel.shape[0] - 1) // 2
    cx = (kernel.shape[1] - 1) // 2
    dy, dx = ky - cy, kx - cx
    keep = (dy != 0) | (dx != 0)
    dy, dx = dy[keep], dx[keep]
    order = np.argsort(dy * dy + dx * dx, kind="stable")
    return dy[order], dx[order]


def _propagate_to_donut(mask, donut_idx, offsets):
    """
    Nearest-nucleus label for each donut pixel.

    Every donut pixel is by construction within the big-kernel footprint of some
    nucleus, so scanning the footprint in order of increasing distance and
    keeping the first hit gives the euclidean nearest label.
    """
    H, W = mask.shape
    flat = mask.ravel()
    y, x = np.divmod(donut_idx, W)
    out = np.zeros(donut_idx.size, dtype=mask.dtype)
    todo = np.arange(donut_idx.size)
    for dy, dx in zip(*offsets):
        if todo.size == 0:
            break
        yy = y[todo] + dy
        xx = x[todo] + dx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        v = np.zeros(todo.size, dtype=mask.dtype)
        v[ok] = flat[yy[ok] * W + xx[ok]]
        hit = v > 0
        out[todo[hit]] = v[hit]
        todo = todo[~hit]
    return out


def _group_median(lab, val, n):
    """Median of `val` grouped by integer `lab` in [0, n)."""
    order = np.lexsort((val, lab))
    lab_s, val_s = lab[order], val[order]
    counts = np.bincount(lab_s, minlength=n)
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    out = np.zeros(n, dtype=np.float64)
    nz = np.flatnonzero(counts)
    if nz.size == 0:
        return out
    mid = starts[nz] + counts[nz] // 2
    m = val_s[mid].astype(np.float64)
    even = (counts[nz] % 2) == 0
    m[even] = 0.5 * (m[even] + val_s[mid[even] - 1])
    out[nz] = m
    return out


# --------------------------------------------------------------------------- #
# per plane / per stack
# --------------------------------------------------------------------------- #
def process_plane(plane, mask, plane_index, offsets,
                  small_kernel=SMALL_KERNEL, big_kernel=BIG_KERNEL,
                  camera_min=100.0, compute_medians=True):
    flat_mask = mask.ravel()
    nuc_idx = np.flatnonzero(flat_mask)
    if nuc_idx.size == 0:
        return None

    W = mask.shape[1]
    raw_flat = plane.ravel()

    occupied = (mask > 0).view(np.uint8)
    donut = (cv2.dilate(occupied, big_kernel).astype(bool)
             & ~cv2.dilate(occupied, small_kernel).astype(bool))
    donut_idx = np.flatnonzero(donut.ravel())
    donut_lab = _propagate_to_donut(mask, donut_idx, offsets).astype(np.int64)

    nuc_lab = flat_mask[nuc_idx].astype(np.int64)
    n = int(nuc_lab.max()) + 1

    nuc_val = raw_flat[nuc_idx].astype(np.float32)
    nuc_val -= camera_min
    np.clip(nuc_val, 0, None, out=nuc_val)
    cyt_val = raw_flat[donut_idx].astype(np.float32)
    cyt_val -= camera_min
    np.clip(cyt_val, 0, None, out=cyt_val)

    counts = np.bincount(nuc_lab, minlength=n)
    nuc_sum = np.bincount(nuc_lab, weights=nuc_val, minlength=n)
    y_sum = np.bincount(nuc_lab, weights=nuc_idx // W, minlength=n)
    x_sum = np.bincount(nuc_lab, weights=nuc_idx % W, minlength=n)
    cyt_cnt = np.bincount(donut_lab, minlength=n)
    cyt_sum = np.bincount(donut_lab, weights=cyt_val, minlength=n)

    labels = np.flatnonzero(counts)
    c = counts[labels].astype(np.float64)
    nuc_mean = nuc_sum[labels] / c
    cy = y_sum[labels] / c
    cx = x_sum[labels] / c
    cc = cyt_cnt[labels]
    cyt_mean = np.where(cc > 0, cyt_sum[labels] / np.maximum(cc, 1), 0.0)

    if compute_medians:
        nuc_med = _group_median(nuc_lab, nuc_val, n)[labels]
        cyt_med = _group_median(donut_lab, cyt_val, n)[labels]
    else:
        nuc_med = np.full(labels.size, np.nan)
        cyt_med = np.full(labels.size, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_mean = np.where(cyt_mean != 0, nuc_mean / cyt_mean, np.nan)
        ratio_med = np.where(cyt_med != 0, nuc_med / cyt_med, np.nan)

    return {
        "Z": np.full(labels.size, plane_index, dtype=np.int32),
        "Label": labels.astype(np.int32),
        "Size": c,
        "CentroidX": cx,
        "CentroidY": cy,
        "MeanIntensityNucleus": nuc_mean,
        "MedianIntensityNucleus": nuc_med,
        "MeanIntensityCytoplasm": cyt_mean,
        "MedianIntensityCytoplasm": cyt_med,
        "NucleusCytoplasmRatioMean": ratio_mean,
        "NucleusCytoplasmRatioMedian": ratio_med,
    }


def measure_stack_nuclear_stats(raw_stack, mask_stack,
                                small_kernel=SMALL_KERNEL,
                                big_kernel=BIG_KERNEL,
                                camera_min=100.0,
                                compute_medians=True):
    offsets = _kernel_offsets(big_kernel)
    chunks = []
    for i, (plane, mask) in enumerate(zip(raw_stack, mask_stack)):
        res = process_plane(plane, mask, i, offsets, small_kernel, big_kernel,
                            camera_min, compute_medians)
        if res is not None:
            chunks.append(res)
    if not chunks:
        return pl.DataFrame({c: [] for c in COLUMNS})
    merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    return pl.DataFrame(merged).select(COLUMNS)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_channel(path, channel):
    """Read a single channel of a (Z, C, Y, X) OME-TIFF without decoding the rest."""
    try:
        with TiffFile(path) as tif:
            series = tif.series[0]
            axes = series.axes
            if "C" in axes and series.ndim >= 3:
                store = series.aszarr()
                import zarr
                arr = zarr.open(store, mode="r")
                sl = [slice(None)] * arr.ndim
                sl[axes.index("C")] = channel
                return np.asarray(arr[tuple(sl)])
    except Exception:
        pass
    return imread(path)[:, channel]

def process_file(raw_path, mask_path, output_path, channel=0,
                 camera_min=100.0, compute_medians=True):
    raw_stack = read_channel(raw_path, channel)
    mask_stack = imread(mask_path)
    df = measure_stack_nuclear_stats(raw_stack, mask_stack,
                                     camera_min=camera_min,
                                     compute_medians=compute_medians)
    df.write_csv(output_path)
    return output_path


def run(raw_paths, mask_paths, output_paths, channel=0, n_jobs=8,
        camera_min=100.0, compute_medians=True):
    """
    One process per file: no array pickling, I/O overlaps with compute.
 
    `return_as` makes joblib yield results as they complete instead of
    returning a finished list, which is what tqdm needs to update live.
    "generator_unordered" needs joblib >= 1.4, "generator" >= 1.3.
    """
    jobs = (delayed(process_file)(r, m, o, channel, camera_min, compute_medians)
            for r, m, o in zip(raw_paths, mask_paths, output_paths))
    kwargs = dict(n_jobs=n_jobs, backend="loky", batch_size=1)
    try:
        parallel = Parallel(return_as="generator_unordered", **kwargs)
    except (TypeError, ValueError):
        try:
            parallel = Parallel(return_as="generator", **kwargs)
        except (TypeError, ValueError):
            parallel = Parallel(**kwargs)  # old joblib: bar will not animate
    with parallel:
        for _ in tqdm(parallel(jobs), total=len(raw_paths), smoothing=0.1):
            pass


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

raw_paths, mask_paths, output_paths = (
    experiment_filemap[raw_dir_name].to_list(),
    experiment_filemap[os.path.basename(mask_dir)].to_list(),
    [os.path.join(output_dir, os.path.basename(r).replace(".ome.tiff", ".csv")) for r in experiment_filemap[raw_dir_name].to_list()]
)

run(raw_paths, mask_paths, output_paths, channel=channel, n_jobs=8)