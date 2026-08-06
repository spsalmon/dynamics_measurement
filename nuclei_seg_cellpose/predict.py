import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
from cellpose.utils import stitch3D
from tqdm import tqdm
import os
import polars as pl
from tifffile import imread, imwrite

from towbintools.foundation.file_handling import get_dir_filemap, read_filemap
from queue import Queue
from threading import Thread
io.logger_setup()

def prefetch_stacks(raw_paths, prefetch=8, channel=None):
    q = Queue(maxsize=prefetch)

    def producer():
        for raw_path in raw_paths:
            raw = imread(raw_path)
            if channel is not None:
                raw = raw[:, channel]
            raw = np.expand_dims(raw, axis=-1)
            q.put((raw_path, raw))
        q.put(None)

    Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        yield item

experiment_dir = "/mnt/towbin.data/shared/spsalmon/20260609_163634_517_ZIVA_60x_307_405_yap_dynamics"
raw_dir = os.path.join(experiment_dir, "raw_stacks")
raw_dir_name = os.path.basename(raw_dir)
analysis_dir = os.path.join(experiment_dir, "analysis_stacks")
report_dir = os.path.join(experiment_dir, "analysis", "report")
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)
channel = 1


experiment_filemap = get_dir_filemap(raw_dir)
experiment_filemap = experiment_filemap.rename({"ImagePath": raw_dir_name})

analysis_filemap_path = [os.path.join(report_dir, f) for f in os.listdir(report_dir) if "analysis_filemap_annotated" in f]
print(analysis_filemap_path)
print(f'Number of files in experiment filemap: {len(experiment_filemap)}')
if len(analysis_filemap_path) > 0:
    analysis_filemap = read_filemap(analysis_filemap_path[0]).select(["Point", "Time", "Ignore"])
    experiment_filemap = experiment_filemap.join(analysis_filemap, on=["Point", "Time"], how="left").filter(~pl.col("Ignore")).drop("Ignore")

print(f'Number of files to process after filtering: {len(experiment_filemap)}')
model = models.CellposeModel(gpu=True, pretrained_model="/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/cellpose/models/emr1_60x")

output_path = os.path.join(analysis_dir, "ch2_seg_cellpose")
os.makedirs(output_path, exist_ok=True)

image_paths = experiment_filemap[raw_dir_name].to_list()
images_to_process = []
for image_path in image_paths:
    output_file_path = os.path.join(output_path, os.path.basename(image_path))
    if not os.path.exists(output_file_path):
        images_to_process.append(image_path)

image_paths = images_to_process

print(f'Processing {len(image_paths)} images with Cellpose ...')

for image_path, image in tqdm(prefetch_stacks(image_paths, channel=channel), total=len(image_paths)):
    print(f'Image shape: {image.shape}')

    masks, _, _ = model.eval(image, z_axis=None, channel_axis=None, do_3D=False, batch_size=128)
    masks = stitch3D(masks, stitch_threshold=0.25)

    imwrite(os.path.join(output_path, os.path.basename(image_path)), masks.astype(np.uint16), compression="zlib")