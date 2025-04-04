import numpy as np

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import normalize

from stardist.models import StarDist2D, Config2D
from stardist import gputools_available
from tifffile import imwrite
import os

import pandas as pd

def class_from_res(res):
    cls_dict = dict((i+1,c) for i,c in enumerate(res['class_id']))
    return cls_dict

def segment_nuclei_panoptic_stardist(image_path:str, model:StarDist2D) -> None:

	CLASS_VALUES = {"background": 0, "epidermis": 1, "intestine": 2, "other": 3, "error": 4}
	CLASS_ID_TO_NAME = {v: k for k, v in CLASS_VALUES.items()}


	nuclei_image = imread(image_path)[:, 1, ...]
	if nuclei_image.ndim > 2:
		# Create an empty array of the same shape as the input image for storing the binary masks of segmented nuclei
		nuclei_mask_stack = np.zeros_like(nuclei_image, dtype="uint8")
		classes_df = pd.DataFrame()
		# Perform nuclei segmentation on each plane in the stack
		for index, plane in enumerate(nuclei_image):
			img = normalize(plane, 1,99.8, axis=(0, 1))
			labels, details = model.predict_instances(img, verbose = False, show_tile_progress=False)
			classes = class_from_res(details)
			
			plane_classes_df = pd.DataFrame(list(classes.items()), columns=['Label', 'ClassID'])
			plane_classes_df['Z'] = index
			plane_classes_df['Class'] = plane_classes_df['ClassID'].map(CLASS_ID_TO_NAME)
			classes_df = pd.concat([classes_df, plane_classes_df])

			# Store the mask in the output array
			nuclei_mask_stack[index, :, :] = (labels).astype(np.uint8)

		print(f'DONE ! {os.path.basename(image_path)}')
		# Save the mask
		imwrite(os.path.join(output_mask_dir, os.path.basename(image_path)), nuclei_mask_stack, compression='zlib')
		classes_df.to_csv(os.path.join(output_class_dir, os.path.basename(image_path).replace('.ome.tiff', '.csv')), index=False)

input_dir = "/mnt/towbin.data/shared/spsalmon/20241115_122955_248_ZIVA_40x_raga1_full_deletion/raw_ometiff/"
output_mask_dir = "/mnt/towbin.data/shared/spsalmon/20241115_122955_248_ZIVA_40x_raga1_full_deletion/analysis/ch2_stardist/"
output_class_dir = "/mnt/towbin.data/shared/spsalmon/20241115_122955_248_ZIVA_40x_raga1_full_deletion/analysis/ch2_nuclei_types_stardist/"

os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_class_dir, exist_ok=True)

use_gpu = gputools_available()

print(f"GPU enabled: {use_gpu}")

images_path = sorted([os.path.join(input_dir, x) for x in os.listdir(input_dir)])
images_path = [x for x in images_path if 'DIA' not in x]

model = StarDist2D(None, name='panoptic_stardist_emr_semi_auto_no_grid', basedir='/mnt/towbin.data/shared/spsalmon/models/stardist/')

for image_path in tqdm(images_path):
	segment_nuclei_panoptic_stardist(image_path, model)