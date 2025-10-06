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
from towbintools.foundation.image_handling import read_tiff_file
import argparse
import yaml
import shutil
import pandas as pd

np.random.seed(42)
lbl_cmap = random_label_cmap()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args


config_file = get_args().config
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

image_directories = config.get("image_directories", None)
mask_directories = config.get("mask_directories", None)
classes_directories = config.get("classes_directories", None)

save_dir = config.get("save_dir", None)
model_name = config.get("model_name", None)
checkpoint_path = config.get("continue_training_from_checkpoint", None)
n_rays = config.get("n_rays", 32)
panoptic = config.get("panoptic", False)
n_classes = config.get("n_classes", None)
max_epochs = config.get("max_epochs", 400)
steps_per_epoch = config.get("steps_per_epoch", 100)
patch_size = config.get("patch_size", 512)
train_val_split_ratio = config.get("train_val_split_ratio", 0.2)
subsample_factor = config.get("subsample_factor", 1)
channels_to_segment = config.get("channels_to_segment", [0])
use_gpu = config.get("use_gpu", True)

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    x = x/np.max(x)
    return x, y


X = [sorted([os.path.join(d, f) for f in os.listdir(d)]) for d in image_directories]
Y = [sorted([os.path.join(d, f) for f in os.listdir(d)]) for d in mask_directories]
X = sum(X, [])
Y = sum(Y, [])

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

assert all(Path(x).stem == Path(y).stem for x, y in zip(X, Y))

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
grid = (subsample_factor, subsample_factor)
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
if panoptic:
    if n_classes is None:
        n_classes = max([max(c.values()) for c in classes_trn])
    print(f"Using {n_classes} classes for panoptic segmentation.")
    conf.n_class = n_classes

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(None, allow_growth=True)

if checkpoint_path is not None:
    print(f"Continuing training from checkpoint: {checkpoint_path}")
    path = Path(checkpoint_path)
    model = StarDist2D(None, name=path.name, basedir=path.parent.absolute())
    shutil.copytree(path, os.path.join(save_dir, model_name), dirs_exist_ok=True)
    model = StarDist2D(None, name=model_name, basedir=save_dir)
    model.config.train_epochs = max_epochs
    model.config.train_steps_per_epoch = steps_per_epoch
    model.config.train_patch_size = train_patch_size
    model.config.grid = grid
    model.config.n_rays = n_rays
    model.config.use_gpu = use_gpu
    model.config.n_channel_in = n_channel
    if panoptic:
        if n_classes is None:
            n_classes = max([max(c.values()) for c in classes_trn])
        print(f"Using {n_classes} classes for panoptic segmentation.")
        model.config.n_class = n_classes
    else:
        model.config.n_class = None
else:
    model = StarDist2D(conf, name=model_name, basedir=save_dir)

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

if panoptic:
    model.train(X_trn, Y_trn, classes = classes_trn, validation_data=(X_val,Y_val), augmenter=augmenter)
else:
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)

model.optimize_thresholds(X_val, Y_val)