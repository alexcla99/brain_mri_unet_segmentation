from utils import info, prepare_dataset
from params import dataset_params, model_params
from brainmri import BrainMri
from model import get_u_net
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, array_to_img
import matplotlib.pyplot as plt
import PIL
import os

WEIGHTS_PATH = os.path.join("out", "train", "brain_segmentation.h5")
DEST_PATH = os.path.join("out", "predict")
FIGSIZE = (15, 7)

if __name__ == "__main__":
    """Make predictions (create masks)."""
    keras.backend.clear_session()
    # Loading the dataset
    info("Loading the dataset")
    val_input_img_paths, val_target_img_paths = prepare_dataset(val_only=True)
    val_gen = BrainMri(
        model_params["batch_size"], dataset_params["img_size"], val_input_img_paths, val_target_img_paths
    )
    # Predicting on the validation dataset
    info("Predicting on the validation dataset")
    model = get_u_net()
    model.load_weights(WEIGHTS_PATH)
    val_preds = model.predict(val_gen)
    # Saving the predictions into jpeg files
    for i in range(len(val_preds)):
        img = PIL.Image.open(val_input_img_paths[i])
        mask = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[img_slice]))
        prediction = np.argmax(val_preds[i], axis=-1)
        prediction = np.expand_dims(prediction, axis=-1)
        prediction = PIL.ImageOps.autocontrast(array_to_img(prediction))
        output_name = (val_input_img_paths[i].split(".")[0]).split("/")[-1]
        output_name += ("_prediction" + dataset_params["data_type"])
        fig = plt.figure(figsize=FIGSIZE)
        data = [img, mask, prediction]
        titles = ["MRI", "Mask", "Predicted mask"]
        for j in range(len(data)):
            fig.add_subplot(1, 3, i + 1)
            plt.imshow(data[j])
            plt.axis("off")
            plt.title(title[j])
        fig.suptitle(output_name)
        plt.savefig(os.path.join(DEST_PATH, output_name))
    # Finishing the program
    info("Done")