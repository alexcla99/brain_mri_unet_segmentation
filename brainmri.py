from params import dataset_params, model_params
from utils import load_dataset
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

class BrainMri(keras.utils.Sequence):
    """Iterative data access class."""

    def __init__(self) -> None:
    	"""Builder."""
    	self.input_img_paths, self.target_img_paths = load_dataset()

    def __len__(self) -> int:
    	"""Length of the dataset for each batch."""
    	return len(self.target_img_paths) // model_params["batch_size"]

    def __getitem__(self, idx:int) -> (np.ndarray, np.ndarray):
    	"""Item getter (MRI and mask)."""
    	i = idx * model_params["batch_size"]
    	batch_input_img_paths = self.input_img_paths[i:i + model_params["batch_size"]]
    	batch_target_img_paths = self.target_img_paths[i:i + model_params["batch_size"]]
    	x = np.zeros((model_params["batch_size"],) + dataset_params["img_size"] + (3,), dtype=np.float32)
    	for j, path in enumerate(batch_input_img_paths):
    		img = load_img(path, target_size=dataset_params["img_size"])
    		x[j] = img
    	y = np.zeros((model_params["batch_size"],) + dataset_params["img_size"] + (1,), dtype=np.float32)
    	for j, path in enumerate(batch_target_img_paths):
    		img = load_img(path, target_size=dataset_params["img_size"], colormode="grayscale")
    		y[j] = np.expand_dims(img, 2)
    		y[j] -= 1
    	return x, y / 255