from utils import info, prepare_dataset#, dice_loss
from params import dataset_params, model_params
from model import get_u_net
from brainmri import BrainMri
from tensorflow import keras
import os

# IMG_SLICE = 4 # DEBUG
OUT_PATH = os.path.join("out", "train")

if __name__ == "__main__":
    """Train the U-Net CNN to segment brain MRIs."""
    keras.backend.clear_session()
    # Defining the train / validation sets
    info("Defining the train / validation sets:")
    train_input_img_paths, train_target_img_paths, val_input_img_paths, val_target_img_paths = prepare_dataset()
    train_gen = BrainMri(
        model_params["batch_size"], dataset_params["img_size"], train_input_img_paths, train_target_img_paths
    )
    val_gen = BrainMri(
        model_params["batch_size"], dataset_params["img_size"], val_input_img_paths, val_target_img_paths
    )
    print("Number of train samples: " + str(train_gen.__len__()))
    print("Number of validation samples: " + str(val_gen.__len__()))
    # print(train_gen.__getitem__(IMG_SLICE)) # DEBUG
    # Compiling the model
    info("Compiling the model")
    model = get_u_net()
    model.compile(
        optimize="Adam",
        loss="categorical_crossentropy" # TODO dice_loss
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(OUT_PATH, "brain_segmentation.h5"), save_best_only=True)
    ]
    # Training the model
    info("Training the model:")
    model.fit(train_gen, epochs=model_params["num_eppochs"], validation_data=val_gen, callbacks=callbacks)
    # Finishing the program
    info("Done")