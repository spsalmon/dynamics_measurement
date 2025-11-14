import numpy as np

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import normalize

from stardist.models import StarDist2D, Config2D
from stardist import gputools_available
from tifffile import imwrite
import os
from csbdeep.data import PercentileNormalizer
import csbdeep
import pandas as pd
from typing import List
from utils import read_tiff_file

def class_from_res(res):
    cls_dict = dict((i+1,c) for i,c in enumerate(res['class_id']))
    return cls_dict

def predict_stardist(image_path:str, output_path:str, model:StarDist2D, channels_to_keep:List[int], normalizer: csbdeep.Normalizer = PercentileNormalizer(1, 99.8, do_after=False), prob_thresh=None) -> None:
	image = read_tiff_file(image_path, channels_to_keep=channels_to_keep)
	nuclei_mask_stack = np.zeros_like(image, dtype="uint16")
	for i, plane in enumerate(image):
		img = normalize(plane, 1,99.8, axis=(0, 1))
		labels, _ = model.predict_instances(img, prob_thresh=prob_thresh, normalizer=normalizer)
		nuclei_mask_stack[i, :, :] = (labels).astype(np.uint16)
	imwrite(output_path, nuclei_mask_stack, compression='zlib')

# def segment_nuclei_panoptic_stardist(image_path:str, model:StarDist2D) -> None:

# 	CLASS_VALUES = {"background": 0, "epidermis": 1, "intestine": 2, "other": 3, "error": 4}
# 	CLASS_ID_TO_NAME = {v: k for k, v in CLASS_VALUES.items()}


# 	nuclei_image = imread(image_path)[:, 1, ...]
# 	if nuclei_image.ndim > 2:
# 		# Create an empty array of the same shape as the input image for storing the binary masks of segmented nuclei
# 		nuclei_mask_stack = np.zeros_like(nuclei_image, dtype="uint8")
# 		classes_df = pd.DataFrame()
# 		# Perform nuclei segmentation on each plane in the stack
# 		for index, plane in enumerate(nuclei_image):
# 			img = normalize(plane, 1,99.8, axis=(0, 1))
# 			labels, details = model.predict_instances(img, verbose = False, show_tile_progress=False)
# 			classes = class_from_res(details)
			
# 			plane_classes_df = pd.DataFrame(list(classes.items()), columns=['Label', 'ClassID'])
# 			plane_classes_df['Z'] = index
# 			plane_classes_df['Class'] = plane_classes_df['ClassID'].map(CLASS_ID_TO_NAME)
# 			classes_df = pd.concat([classes_df, plane_classes_df])

# 			# Store the mask in the output array
# 			nuclei_mask_stack[index, :, :] = (labels).astype(np.uint8)

# 		print(f'DONE ! {os.path.basename(image_path)}')
# 		# Save the mask
# 		imwrite(os.path.join(output_mask_dir, os.path.basename(image_path)), nuclei_mask_stack, compression='zlib')
# 		classes_df.to_csv(os.path.join(output_class_dir, os.path.basename(image_path).replace('.ome.tif', '.csv')), index=False)

input_dir = "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/stardist/emr1_panoptic_dataset_60x_auto_seg/raw"
output_dir = "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/stardist/emr1_panoptic_dataset_60x_auto_seg/mask"
os.makedirs(output_dir, exist_ok=True)
model = StarDist2D(None, name='emr1_60x', basedir='/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/stardist/')
prob_thresh = 0.6
channels_to_keep = [1]
image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
output_paths = [os.path.join(output_dir, os.path.basename(f)) for f in image_paths]
image_paths.sort()
output_paths.sort()

for img_path, out_path in tqdm(zip(image_paths, output_paths), total=len(image_paths)):
	if model.n_classes is None or model.n_classes == 1:
		predict_stardist(img_path, out_path, model, channels_to_keep, prob_thresh=prob_thresh)
