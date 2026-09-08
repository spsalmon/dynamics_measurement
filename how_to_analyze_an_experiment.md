# How to Analyze Signaling Dynamics Experiments

## 1. Segment the nuclei : 

1. Train a cellpose model for nuclei segmentation or pick an already trained model.
2. Reserve some GPUs with salloc
3. Run the python script at : nuclei_seg/predict.py (first modify your desired input and output)

## 2. Classify the nuclei type

1. Train a standard towbintools segmentation model for nuclei classification using the napari plugin and the standard scripts.
2. Run a pipeline with segmentation using this model on your stacks

## 3. Combine segmentation and classification

1. Run the script : nuclei_seg/combine_seg_and_type.py (first modify your desired input and output)

## 4. Measure the intensities and other features

1. Run the script : fluo_measurement/measure.py (first modify your desired input and output)