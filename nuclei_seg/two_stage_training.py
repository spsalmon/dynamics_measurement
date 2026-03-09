import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D
import yaml
import shutil
import pandas as pd
from utils import read_tiff_file, copy_without_permissions, get_args, augmenter
from tifffile import imwrite
import csbdeep.data
from csbdeep.data import PercentileNormalizer
from typing import List
from skimage.measure import label, regionprops
from joblib import Parallel, delayed
from tensorflow.keras.utils import PyDataset
from tensorflow.keras.utils import Sequence
from functools import lru_cache
from collections import OrderedDict

def predict_stardist(image_path:str, output_path:str, model:StarDist2D, channels_to_keep:List[int], normalizer: csbdeep.data.Normalizer = PercentileNormalizer(1, 99.8, do_after=False), prob_thresh=None) -> None:
	try:
		image = read_tiff_file(image_path, channels_to_keep=channels_to_keep)
		nuclei_mask_stack = np.zeros_like(image, dtype="uint16")
		for i, plane in enumerate(image):
			labels, _ = model.predict_instances(plane, prob_thresh=prob_thresh, normalizer=normalizer)
			nuclei_mask_stack[i, :, :] = (labels).astype(np.uint16)
		imwrite(output_path, nuclei_mask_stack, compression='zlib')
	except Exception as e:
		print(f"Error processing {image_path}: {e}")

np.random.seed(42)
lbl_cmap = random_label_cmap()

config_file = get_args().config
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

image_directories = config.get("image_directories", None)
mask_directories = config.get("mask_directories", None)

save_dir = config.get("save_dir", None)
model_name = config.get("model_name", None)
checkpoint_path = config.get("continue_training_from_checkpoint", None)
pretrained_model = config.get("pretrained_model", None)
skip_seg_training_and_load = config.get("skip_seg_training_and_load", None)
n_rays = config.get("n_rays", 32)
panoptic = config.get("panoptic", False)
n_classes = config.get("n_classes", None)
max_epochs = config.get("max_epochs", 400)
steps_per_epoch = config.get("steps_per_epoch", -1)
patch_size = config.get("patch_size", 512)
train_val_split_ratio = config.get("train_val_split_ratio", 0.2)
downsample_factor = config.get("downsample_factor", 1)
channels_to_segment = config.get("channels_to_segment", [0])
use_gpu = config.get("use_gpu", True)
batch_size = config.get("batch_size", 4)
classification_max_epochs = config.get("classification_max_epochs", 200)
classification_training_patch_size = config.get("classification_training_patch_size", 1024)
classification_training_steps = config.get("classification_training_steps", 500)

use_gpu = use_gpu and gputools_available()

normalizer = PercentileNormalizer(1, 99.8, do_after=False)

if skip_seg_training_and_load is None:
    X = [sorted([os.path.join(d, f) for f in os.listdir(d)]) for d in image_directories]
    Y = [sorted([os.path.join(d, f) for f in os.listdir(d)]) for d in mask_directories]
    X = sum(X, [])
    Y = sum(Y, [])

    assert all(Path(x).stem == Path(y).stem for x, y in zip(X, Y))
    X = [read_tiff_file(f, channels_to_keep=channels_to_segment) for f in tqdm(X)]
    Y = [imread(f) for f in tqdm(Y)]

    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    axis_norm = (-2, -1)   # normalize channels independently
    # axis_norm = (-3, -2, -1) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    # X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    X = [normalizer(x.astype(np.float32), axes="YX") for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(train_val_split_ratio * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]


    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

    # generate fake classes
    classes = []
    for y in Y:
        class_dict = {}
        labels = np.unique(y)
        for lbl in labels:
            if lbl == 0:
                continue
            class_dict[lbl] = np.random.randint(1, 3)  # assign random class 1 or 2
        classes.append(class_dict)

    assert len(X) == len(Y) == len(classes)
    classes_trn, classes_val = [classes[i] for i in ind_train], [classes[i] for i in ind_val]

    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    print(f'Using {n_rays} rays and {"GPU" if use_gpu else "CPU"}.')

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (downsample_factor, downsample_factor)
    train_patch_size = (patch_size, patch_size)

    if steps_per_epoch == -1:
        steps_per_epoch = len(X_trn) // batch_size

    if n_classes is None:
        n_classes = max([max(c.values()) for c in classes_trn])

    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = n_channel,
        train_epochs = max_epochs,
        train_steps_per_epoch = steps_per_epoch,
        train_patch_size = train_patch_size,
        train_batch_size = batch_size,
        n_classes    = n_classes,
        train_loss_weights = (1.0, 1.0, 0.0) if panoptic else (1.0, 1.0),
    )

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(None, allow_growth=True)


    if checkpoint_path is not None:
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

    # freeze the classification head of the network during the first stage
    keras_model = model.keras_model

    keras_model.get_layer(name='features_class').trainable = False
    keras_model.get_layer(name='prob_class').trainable = False

    print("#### TRAINING THE SEGMENTATION PART OF THE NETWORK ####")

    model.train(X_trn, Y_trn, classes = classes_trn, validation_data=(X_val,Y_val,classes_val), augmenter=augmenter)
    model.optimize_thresholds(X_val, Y_val)
else:
    print(f"Skipping segmentation training and loading model from {skip_seg_training_and_load}")
    model_path = Path(skip_seg_training_and_load)
    model = StarDist2D(None, name=model_path.name, basedir=model_path.parent.absolute())


print("#### SEGMENTATION PART TRAINED ####")
print("---------------------------------")
print("#### USING THE TRAINED MODEL TO PREDICT MASKS FOR THE SECOND STAGE ####")

segmentation_source = config.get("segmentation_source", None)
output_mask_directory = config.get("output_mask_directory", None)
output_classes_directory = config.get("output_classes_directory", None)

raw_planes_dir = os.path.normpath(segmentation_source) + "_planes"
mask_planes_dir = os.path.normpath(output_mask_directory) + "_planes"

os.makedirs(raw_planes_dir, exist_ok=True)
os.makedirs(mask_planes_dir, exist_ok=True)

segmentation_channel = config.get("segmentation_channel", None)
tissue_marker_channel = config.get("tissue_marker_channel", None)
marker_intensity_threshold = config.get("marker_intensity_threshold", None)
min_n_nuclei = config.get("min_n_nuclei", 5)
rerun_segmentation = config.get("rerun_segmentation", True)
rerun_splitting = config.get("rerun_splitting", True)

if segmentation_source is None:
    raise ValueError("segmentation_source must be provided to generate masks for the second stage")

if output_mask_directory is None or output_classes_directory is None:
    raise ValueError("output_mask_directory and output_classes_directory must be provided to generate masks for the second stage")

os.makedirs(output_mask_directory, exist_ok=True)
os.makedirs(output_classes_directory, exist_ok=True)

image_paths = sorted([os.path.join(segmentation_source, f) for f in os.listdir(segmentation_source)])
output_paths = sorted([os.path.join(output_mask_directory, os.path.basename(f)) for f in image_paths])

for img_path, out_path in tqdm(zip(image_paths, output_paths), total=len(image_paths)):
    if not os.path.exists(out_path) or rerun_segmentation:
        predict_stardist(img_path, out_path, model, segmentation_channel, normalizer=normalizer)

# extract planes with enough nuclei and save corresponding class csv files
def process_img_and_masks(img_path, mask_path, segmentation_channel, tissue_marker_channel, marker_intensity_threshold, min_n_nuclei, output_classes_directory, rerun=False):
    
    if not rerun and len([f for f in os.listdir(raw_planes_dir) if Path(img_path).stem in f]) > 0:
        return
    
    img = read_tiff_file(img_path)
    mask = imread(mask_path)
    
    image_name = Path(img_path).stem
    img_marker = img[:, tissue_marker_channel[0]]
    img_seg = img[:, segmentation_channel[0]]

    for i, (img_seg_plane, img_marker_plane, mask_plane) in enumerate(zip(img_seg, img_marker, mask)):
        plane_info = []
        plane_name = f"{image_name}_plane_{i}"
        mask_plane = label(mask_plane)

        n_nuclei = len(np.unique(mask_plane)) - 1  # exclude background
        if n_nuclei < min_n_nuclei:
            continue

        props = regionprops(mask_plane, intensity_image=img_marker_plane)
        for prop in props:
            mean_intensity = prop.intensity_mean
            if mean_intensity >= marker_intensity_threshold:
                plane_info.append({'Label': prop.label, 'Class': 1})  # class 1 for marker positive
            else:
                plane_info.append({'Label': prop.label, 'Class': 2})  # class 2 for marker negative


        plane_info = pd.DataFrame.from_records(plane_info)

        imwrite(os.path.join(raw_planes_dir, f"{plane_name}.tiff"), img_seg_plane, compression='zlib')
        imwrite(os.path.join(mask_planes_dir, f"{plane_name}.tiff"), mask_plane.astype(np.uint8), compression='zlib')
        plane_info.to_csv(os.path.join(output_classes_directory, f"{plane_name}.csv"), index=False)

Parallel(n_jobs=-1)(delayed(process_img_and_masks)(
        img_path,
        mask_path, 
        segmentation_channel, 
        tissue_marker_channel, 
        marker_intensity_threshold, 
        min_n_nuclei, 
        output_classes_directory,
        rerun=rerun_splitting,
    ) for img_path, mask_path in tqdm(zip(image_paths, output_paths), total=len(image_paths)))

print("---------------------------------")
print("#### TRAINING THE CLASSIFICATION PART OF THE NETWORK ####")
# freeze all layers of the network
keras_model = model.keras_model
for layer in keras_model.layers:
    layer.trainable = False

# unfreeze the classification head of the network
keras_model.get_layer(name='features_class').trainable = True
keras_model.get_layer(name='prob_class').trainable = True

model.config.train_loss_weights = (0.0, 0.0, 1.0)

class SimpleDataset(Sequence):
    def __init__(self, list_paths: List[str], normalizer=None, verbose=False):
        self.list_paths = list_paths
        self.normalizer = normalizer
        self.x = np.arange(len(list_paths))
        self.verbose = verbose

    def __len__(self):
        return len(self.list_paths)

    def _load_item(self, path: str):
        img = read_tiff_file(path)
        if self.normalizer is not None:
            img = self.normalizer(img, axes="YX")
        return img

    def __getitem__(self, idx: int):
        if self.verbose:
            print(f"Loading item {idx+1}/{len(self.list_paths)}: {self.list_paths[idx]}")
        img = self._load_item(self.list_paths[idx])
        return img
        
plane_image_paths = sorted([os.path.join(raw_planes_dir, f) for f in os.listdir(raw_planes_dir)])
plane_mask_paths = sorted([os.path.join(mask_planes_dir, f) for f in os.listdir(mask_planes_dir)])
plane_class_paths = sorted([os.path.join(output_classes_directory, f) for f in os.listdir(output_classes_directory)])

rng = np.random.RandomState(42)
ind = rng.permutation(len(plane_image_paths))
n_val = max(1, int(round(train_val_split_ratio * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]

train_images, val_images = [plane_image_paths[i] for i in ind_train], [plane_image_paths[i] for i in ind_val]
train_masks, val_masks = [plane_mask_paths[i] for i in ind_train], [plane_mask_paths[i] for i in ind_val]
train_classes, val_classes = [plane_class_paths[i] for i in ind_train], [plane_class_paths[i] for i in ind_val]

train_images_ds = SimpleDataset(train_images, normalizer=normalizer)
val_images_ds = SimpleDataset(val_images, normalizer=normalizer)
train_masks_ds = SimpleDataset(train_masks, normalizer=None)
val_masks_ds = SimpleDataset(val_masks, normalizer=None)

def load_classes(paths):
    result = []
    for p in paths:
        df = pd.read_csv(p)
        result.append(dict(zip(df['Label'], df['Class'])))
    return result

train_classes_list = load_classes(train_classes)
val_classes_list = load_classes(val_classes)

model.config.train_epochs = classification_max_epochs
model.config.train_steps_per_epoch = classification_training_steps
model.config.train_patch_size = (classification_training_patch_size, classification_training_patch_size)
model.config.train_sample_cache = False
model.config.train_n_val_patches = 128

model.train(train_images_ds, train_masks_ds, classes=train_classes_list, validation_data=(val_images_ds, val_masks_ds, val_classes_list), augmenter=augmenter)

print("#### CLASSIFICATION PART TRAINED ####")
print("#### TWO STAGE TRAINING COMPLETE ####")
