from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
# matplotlib.rcParams["image.interpolation"] = None

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D

np.random.seed(42)
lbl_cmap = random_label_cmap()

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

X = sorted(glob('/mnt/towbin.data/shared/spsalmon/20241014_153459_824_ZIVA_40x_443_training_db/training_db/processed_img/*.tiff'))
Y = sorted(glob('/mnt/towbin.data/shared/spsalmon/20241014_153459_824_ZIVA_40x_443_training_db/training_db/processed_annotations/*.tiff'))

# classes = sorted(glob('/mnt/towbin.data/shared/spsalmon/443 test/split_dataset/classes/*.csv'))
# print(len(classes))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

X = list(map(imread,X))
X = [x[1] for x in X]
Y = list(map(imread,Y))
# classes = [pd.read_csv(c) for c in classes]

## convert classes to dictionary
# classes = [dict(zip(c['label'], c['class'])) for c in classes]
# print(classes)

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
X = [x/np.max(x) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
# print(f'number of images: {len(X)}, number_of masks {len(Y)}, number of classes: {len(classes)}')
print(f'number of images: {len(X)}, number_of masks {len(Y)}')
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.2 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
# X_val, Y_val, classes_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val], [classes[i] for i in ind_val]
# X_trn, Y_trn, classes_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] , [classes[i] for i in ind_train]

X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()

print(f'Using {n_rays} rays and {"GPU" if use_gpu else "CPU"}.')

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2, 2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
    train_epochs = 400,
    train_steps_per_epoch = 100,
    train_patch_size = (512, 512),
    # n_classes = 3,
)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    # limit_gpu_memory(0.8)
    # alternatively, try this:
    limit_gpu_memory(None, allow_growth=True)

model = StarDist2D(conf, name='stardist_fully_manual_annotations', basedir='models')
# model = StarDist2D(None, name='stardist_fully_manual_annotations_subsample', basedir='models')

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

# model.train(X_trn, Y_trn, classes = classes_trn, validation_data=(X_val,Y_val,classes_val), augmenter=augmenter)
model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)

model.optimize_thresholds(X_val, Y_val)