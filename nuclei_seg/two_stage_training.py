from __future__ import print_function, unicode_literals, absolute_import, division
import os
import sys
import numpy as np
# matplotlib.rcParams["image.interpolation"] = None

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D
import argparse
import yaml
import shutil
import pandas as pd
from typing import Optional
from utils import read_tiff_file, copy_without_permissions, get_args, augmenter

np.random.seed(42)
lbl_cmap = random_label_cmap()

config_file = get_args().config
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

image_directories = config.get("image_directories", None)
mask_directories = config.get("mask_directories", None)
classes_directories = config.get("classes_directories", None)

save_dir = config.get("save_dir", None)
model_name = config.get("model_name", None)
checkpoint_path = config.get("continue_training_from_checkpoint", None)
pretrained_model = config.get("pretrained_model", None)
n_rays = config.get("n_rays", 32)
panoptic = config.get("panoptic", False)
n_classes = config.get("n_classes", None)
max_epochs = config.get("max_epochs", 400)
steps_per_epoch = config.get("steps_per_epoch", 100)
patch_size = config.get("patch_size", 512)
train_val_split_ratio = config.get("train_val_split_ratio", 0.2)
downsample_factor = config.get("downsample_factor", 1)
channels_to_segment = config.get("channels_to_segment", [0])
use_gpu = config.get("use_gpu", True)

use_gpu = use_gpu and gputools_available()

if not panoptic:
    print('This script is intended for two-stage training for panoptic segmentation. Set "panoptic": true in the config file.')
    exit(1)

X = [sorted([os.path.join(d, f) for f in os.listdir(d)]) for d in image_directories]
Y = [sorted([os.path.join(d, f) for f in os.listdir(d)]) for d in mask_directories]
X = sum(X, [])
Y = sum(Y, [])

assert all(Path(x).stem == Path(y).stem for x, y in zip(X, Y))

print(f'Loading {len(X)} images and {len(Y)} masks')
X = [read_tiff_file(f, channels_to_keep=channels_to_segment) for f in tqdm(X)]
Y = [imread(f) for f in tqdm(Y)]

if panoptic:
    if classes_directories is None:
        raise ValueError("classes_directories must be provided for panoptic segmentation")
    classes = [sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')]) for d in classes_directories]
    classes = sum(classes, [])
    classes = [pd.read_csv(c) for c in classes]
    classes = [dict(zip(c['label'], c['class'])) for c in classes]

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

axis_norm = (-2, -1)   # normalize channels independently
# axis_norm = (-3, -2, -1) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(train_val_split_ratio * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]


X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

if panoptic:
    assert len(X) == len(Y) == len(classes)
    classes_trn, classes_val = [classes[i] for i in ind_train], [classes[i] for i in ind_val]

print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

print(f'Using {n_rays} rays and {"GPU" if use_gpu else "CPU"}.')

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (downsample_factor, downsample_factor)
train_patch_size = (patch_size, patch_size)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
    train_epochs = max_epochs,
    train_steps_per_epoch = steps_per_epoch,
    train_patch_size = train_patch_size,
)

if n_classes is None:
    n_classes = max([max(c.values()) for c in classes_trn])
print(f"Using {n_classes} classes for panoptic segmentation.")
conf.n_classes = n_classes

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(None, allow_growth=True)


elif checkpoint_path is not None:
    print(f"Continuing training from checkpoint: {checkpoint_path}")

    # check if the checkpoint is compatible with the current configuration
    path = Path(checkpoint_path)
    model = StarDist2D(None, name=path.name, basedir=path.parent.absolute())
    if model.config.n_rays != n_rays:
        raise ValueError(f"Number of rays in the checkpoint ({model.config.n_rays}) does not match the number of rays in the config file ({n_rays}).")
    if model.config.grid != grid:
        raise ValueError(f"Grid in the checkpoint ({model.config.grid}) does not match the grid in the config file ({grid}).")
    shutil.copytree(path, os.path.join(save_dir, model_name), dirs_exist_ok=True, copy_function=copy_without_permissions)
    model = StarDist2D(None, name=model_name, basedir=save_dir)
    # remove the config.json file to avoid confusion
    os.remove(os.path.join(save_dir, model_name, 'config.json'))

    model.config.train_epochs = max_epochs
    model.config.train_steps_per_epoch = steps_per_epoch
    model.config.train_patch_size = train_patch_size
    model.config.grid = grid
    model.config.n_rays = n_rays
    model.config.use_gpu = use_gpu
    model.config.n_channel_in = n_channel
    model.config.n_classes = n_classes
else:
    model = StarDist2D(conf, name=model_name, basedir=save_dir)

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

print("#### TRAINING THE SEGMENTATION PART OF THE NETWORK ####")

model.train(X_trn, Y_trn, classes = classes_trn, validation_data=(X_val,Y_val), augmenter=augmenter)

print("#### SEGMENTATION PART TRAINED ####")
print("---------------------------------")
print("#### USING THE TRAINED MODEL TO PREDICT MASKS FOR THE SECOND STAGE ####")
print("---------------------------------")
print("#### TRAINING THE CLASSIFICATION PART OF THE NETWORK ####")


# model.optimize_thresholds(X_val, Y_val)