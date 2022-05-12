from params import dataset_params
import numpy as np
import tensorflow as tf
import os, random

SRC_PATH = "prepared_dataset"
MASKS_PATH = "masks"
RANDOM_SEED = 1337

def info(s:str) -> None:
	"""Display a string as a debug information."""
	print("[INFO] " + s)

def load_dataset() -> (list, list):
    """Prepare the dataset and its masks."""
    input_img_paths = sorted([
    	os.path.join(SRC_PATH, filename)
    	for filename in os.listdir(SRC_PATH)
    	if filename.endswith(dataset_params["data_type"])
    ])
    target_img_paths = sorted([
    	os.path.join(SRC_PATH, MASKS_PATH, filename)
    	for filename in os.listdir(os.path.join(SRC_PATH, MASKS_PATH))
    	if filename.endswith(dataset_params["data_type"]) and not filename.startswith(".")
    ])
    print("Number of samples: %d" % len(input_img_paths))
    return input_img_paths, target_img_paths

def prepare_dataset(val_only:bool=False) -> (list, list) or (list, list, list, list):
	"""Prepare the train and validation datasets."""
	input_img_paths, target_img_paths = load_dataset()
	random.Random(RANDOM_SEED).shuffle(input_img_paths)
	random.Random(RANDOM_SEED).shuffle(target_img_paths)
	train_input_img_paths = input_img_paths[:-dataset_params["val_samples"]]
	train_target_img_paths = target_img_paths[:-dataset_params["val_samples"]]
	val_input_img_paths = input_img_paths[-dataset_params["val_samples"]:]
	val_target_img_paths = target_img_paths[-dataset_params["val_samples"]:]
	if val_only == True:
		return val_input_img_paths, val_target_img_paths
	return train_input_img_paths, train_target_img_paths, val_input_img_paths, val_target_img_paths

def dice_loss(y_true:np.ndarray, y_pred:np.ndarray) -> float:
	"""Function to compute Dice loss."""
	smooth = 1.
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.math.sigmoid(y_pred)
	numerator = 2. * tf.reduce_sum(y_true * y_pred) + smooth
	denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
	return 1. - numerator / denominator