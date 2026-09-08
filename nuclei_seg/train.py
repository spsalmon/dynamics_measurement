from __future__ import print_function, unicode_literals, absolute_import, division
from cellpose import io, models, train
import os
# matplotlib.rcParams["image.interpolation"] = None
import yaml
from utils import read_tiff_file, get_args
from sklearn.model_selection import train_test_split
from shutil import copyfile
from tifffile import imwrite
io.logger_setup()

config_file = get_args().config
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

image_directories = config.get("image_directories", None)
mask_directories = config.get("mask_directories", None)
save_dir = config.get("save_dir", None)
model_name = config.get("model_name", None)
max_epochs = config.get("max_epochs", 400)
batch_size = config.get("batch_size", 4)
train_val_split_ratio = config.get("train_val_split_ratio", 0.2)
channels_to_segment = config.get("channels_to_segment", [0])
use_gpu = config.get("use_gpu", True)

# first, we need to split our data into training and testing sets
train_dir = os.path.join(save_dir, "train")
test_dir = os.path.join(save_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

image_paths = []
mask_paths = []
for img_dir, mask_dir in zip(image_directories, mask_directories):
    images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
    images.sort()
    masks.sort()
    image_paths.extend(images)
    mask_paths.extend(masks)

train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=train_val_split_ratio, random_state=42)

for img_path, mask_path in zip(train_image_paths, train_mask_paths):
    img = read_tiff_file(img_path, channels_to_keep = channels_to_segment)
    
    imwrite(os.path.join(train_dir, os.path.basename(img_path)), img)
    mask_output_path = os.path.join(train_dir, os.path.basename(mask_path))
    mask_extension = os.path.splitext(mask_path)[1]
    mask_output_path = mask_output_path.replace(mask_extension, "_mask.tiff")
    copyfile(mask_path, mask_output_path)

for img_path, mask_path in zip(test_image_paths, test_mask_paths):
    img = read_tiff_file(img_path, channels_to_keep = channels_to_segment)
    
    imwrite(os.path.join(test_dir, os.path.basename(img_path)), img)
    mask_output_path = os.path.join(test_dir, os.path.basename(mask_path))
    mask_extension = os.path.splitext(mask_path)[1]
    mask_output_path = mask_output_path.replace(mask_extension, "_mask.tiff")
    copyfile(mask_path, mask_output_path)

output = io.load_train_test_data(train_dir, test_dir, image_filter="",
                                mask_filter="_mask", look_one_level_down=False)
images, labels, image_names, test_images, test_labels, image_names_test = output

model = models.CellposeModel(gpu=use_gpu)

model_path, train_losses, test_losses = train.train_seg(model.net,
                            train_data=images, train_labels=labels,
                            test_data=test_images, test_labels=test_labels,
                            weight_decay=0.1, learning_rate=1e-5,
                            n_epochs=max_epochs, save_path=save_dir, model_name=model_name, batch_size=batch_size)