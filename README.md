# Implementation of a U-Net CNN for brain MRIs segmentation.

Author: alexcla99  
Version: 3.0.0

### Folder content:

```
+-+- out/               # The folder containing outputs
| +--- predict/         # The predictions made by the model
| +--- samples/         # The plotted samples
| +--- train/           # The results obtained during the training phase
|
+-+- prepared_dataset/  # The folder containing the dataset and its masks
| +-+- masks/           # The folder containing the masks
|
+--- __init__.py        # An empty file to make this directory being a Python library
+--- brainmri.py        # A class to iterate over the dataset and its masks
+--- model.py           # The model to be trained
+--- params.py          # The params of the model and the dataset
+--- plot_sample.py     # A script to plot a sample of the dataset (with its associated mask)
+--- predict.py         # A script to make predictions over the validation dataset
+--- README.md          # This file
+--- requirements.txt   # The Python libraries to be installed in order to run the project
+--- summarize_model.py # A script to print the model summary (Keras)
+--- train.py           # A script to train the model
+--- utils.py           # Some utils
```

### Usage:

This library has been implemented and used with Python>=3.8.0

Requirements:
```Shell
pip3 install -r requirements
```

Summarize the model:
```Shell
python3 summarize_model.py
```

Plot a sample of the dataset:
```Shell
Usage: python3 plot_sample.py <slice:int>
# Example: python3 plot_sample.py 4
```
The slice is selected from the "prepared_dataset" folder and the results are saved in the "out/samples" folder.

Train the model:
```Shell
python3 train.py
```
The data to be used is selected from the "prepared_dataset" folder and the results are saved in the "out/train" folder.

Use the model to make predictions:
```Shell
python3 predict.py
```
The data to be used is selected from the "prepared_dataset" folder and the results are saved in the "out/predict" folder.

### Thanks to:

Many thanks to [Buda et al.](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
