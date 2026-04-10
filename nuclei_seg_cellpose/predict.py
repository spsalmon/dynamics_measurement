import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
from tqdm import tqdm
import os
from tifffile import imread, imwrite

from towbintools.foundation.file_handling import get_dir_filemap
from queue import Queue
from threading import Thread
io.logger_setup()

def prefetch_stacks(raw_paths, prefetch=8, channel=1):
    q = Queue(maxsize=prefetch)

    def producer():
        for raw_path in raw_paths:
            raw = imread(raw_path)[:, channel]
            raw = np.expand_dims(raw, axis=-1)
            q.put((raw_path, raw))
        q.put(None)

    Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        yield item

experiment_dir = "/mnt/towbin.data/shared/nschoonjans/20260227_Ziva_60X_405_EV-eat-6RNAi"
raw_dir = os.path.join(experiment_dir, "raw_stacks")
raw_dir_name = os.path.basename(raw_dir)
analysis_dir = os.path.join(experiment_dir, "analysis_stacks")
report_dir = os.path.join(analysis_dir, "report")
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

experiment_filemap = get_dir_filemap(raw_dir)
experiment_filemap = experiment_filemap.rename({"ImagePath": raw_dir_name})

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

for image_path, image in tqdm(prefetch_stacks(image_paths, channel=1), total=len(image_paths)):

    masks, _, _ = model.eval(image, z_axis=None, channel_axis=None, do_3D=False, batch_size=128)

    imwrite(output_file_path, masks.astype(np.uint16), compression="zlib")