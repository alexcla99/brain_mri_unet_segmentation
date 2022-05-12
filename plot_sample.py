from utils import prepare_dataset
from params import dataset_params
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import PIL
import os, sys

FIGSIZE = (10,7)
DEST_PATH = os.path.join("out", "samples")

if __name__ == "__main__":
    """Load a chosen MRI and plot it with its mask into a jpeg file."""
    if len(sys.argv) != 2:
        print("Usage: python3 plot_sample.py <slice:int>")
        print("Example: python3 plot_sample.py 4")
    else:
        img_slice = int(sys.argv[1])
        try:
            # Loading the dataset
            input_img_paths, target_img_paths = prepare_dataset()
            # Loading the slice and its mask
            img = PIL.Image.open(input_img_paths[img_slice])
            mask = PIL.ImageOps.autocontrast(load_img(target_img_paths[img_slice]))
            data = [img, mask]
            titles = ["MRI", "Mask"]
            # Preparing the output
            output_name = (input_img_paths[img_slice].split(".")[0]).split("/")[-1]
            output_name += ("_sample" + dataset_params["data_type"])
            fig = plt.figure(figsize=FIGSIZE)
            for i in range(len(data)):
                fig.add_subplot(1, 2, i + 1)
                plt.imshow(data[i])
                plt.axis("off")
                plt.title(titles[i])
            fig.suptitle(output_name)
            # Saving the image and exiting
            plt.savefig(os.path.join(DEST_PATH, output_name))
            print("Saved as: " + output_name)
        except:
            print(str(sys.exc_info()[1]))