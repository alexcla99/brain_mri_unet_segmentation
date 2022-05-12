from params import dataset_params, model_params
from tensorflow.keras import layers
from tensorflow import keras

def conv_block(x:layers.Layer, filters:int) -> layers.Layer:
    """Convolutional block of the CNN."""
    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=.1)(x)
    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=.1)(x)
    return x

def get_u_net() -> keras.Model:
    """Build a U-Net-like CNN in order to segment MRIs."""
    inputs = keras.Input(shape=dataset_params["img_size"] + (3,), batch_size=model_params["batch_size"])
    # First conv layers
    conv1 = conv_block(inputs, model_params["filters"][0])
    pool1 = layers.MaxPooling2D(2, strides=2, padding="same")(conv1)
    # Second conv layer
    conv2 = conv_block(pool1, model_params["filters"][1])
    pool2 = layers.MaxPooling2D(2, strides=2, padding="same")(conv2)
    # Third conv layer
    conv3 = conv_block(pool2, model_params["filters"][2])
    pool3 = layers.MaxPooling2D(2, strides=2, padding="same")(conv3)
    # Fourth conv layer
    conv4 = conv_block(pool3, model_params["filters"][3])
    pool4 = layers.MaxPooling2D(2, strides=2, padding="same")(conv4)
    # Fifth conv layer (with bottleneck)
    conv5 = conv_block(pool4, model_params["filters"][4])
    # First deconv layer
    deconv4 = layers.Conv2DTranspose(model_params["filters"][3], 2, strides=(2, 2), padding="same")(conv5)
    cat4 = layers.Concatenate(axis=3)([conv4, deconv4])
    conv4 = conv_block(cat4, model_params["filters"][3])
    # Second deconv layer
    deconv3 = layers.Conv2DTranspose(model_params["filters"][2], 2, strides=(2, 2), padding="same")(conv4)
    cat3 = layers.Concatenate(axis=3)([conv3, deconv3])
    conv3 = conv_block(cat3, model_params["filters"][2])
    # Third deconv layer
    deconv2 = layers.Conv2DTranspose(model_params["filters"][1], 2, strides=(2, 2), padding="same")(conv3)
    cat2 = layers.Concatenate(axis=3)([conv2, deconv2])
    conv2 = conv_block(cat2, model_params["filters"][1])
    # Fourth (and last) deconv layer
    deconv1 = layers.Conv2DTranspose(model_params["filters"][0], 2, strides=(2, 2), padding="same")(conv2)
    cat1 = layers.Concatenate(axis=3)([conv1, deconv1])
    conv1 = conv_block(cat1, model_params["filters"][0])
    # Output
    outputs = layers.Conv2D(model_params["num_classes"], 3, strides=1, activation="sigmoid", padding="same")(conv1)
    model = keras.Model(inputs, outputs)
    return model