import tensorflow as tf
import tensorflow.keras as kr

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import ReLU, Lambda, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

NoNormalization = lambda: Lambda(lambda x: x)

AVAILABLE_ARCHS = ["AlexNet", "VGG16", "VGG19", "ResNet50"]

def AlexNet(input_tensor=None, classes=1000, Normalization=NoNormalization):
    x = Conv2D(96, 11, padding="same", activation="relu")(input_tensor)
    x = Normalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, 5, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(384, 3, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = Conv2D(384, 3, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = Conv2D(256, 3, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(classes, activation="softmax")(x)

    return Model(inputs=input_tensor, outputs=x)

def show_available():
    print("Available architectures:", ", ".join(AVAILABLE_ARCHS))

def get_arch(arg, input_shape, classes, **kwargs):
    input_tensor = Input(shape=input_shape)

    if "Normalization" in kwargs:
        if kwargs["Normalization"] == "BatchNormalization":
            kwargs["Normalization"] = BatchNormalization
        elif kwargs["Normalization"] == "LayerNormalization":
            kwargs["Normalization"] = LayerNormalization
        elif kwarfs["Normalization"] == "NoNormalization":
            kwargs["Normalization"] = NoNormalization
        # if its not a string assume its a normalization layer
        elif type(kwargs["Normalization"]) == str:
            print("Warning: couldn't understand your normalization")
            kwargs["Normalization"] = NoNormalization

    if arg == "AlexNet":
        return AlexNet(input_tensor=input_tensor, classes=classes, **kwargs)
    elif arg == "VGG16":
        return VGG16(input_tensor=input_tensor, classes=classes, weights=None, **kwargs)
    elif arg == "VGG19":
        return VGG19(input_tensor=input_tensor, classes=classes, weights=None, **kwargs)
    elif arg == "ResNet50":
        return ResNet50(input_tensor=input_tensor, classes=classes, weights=None, **kwargs)
    elif arg not in AVAILABLE_DATASETS:
        show_available()
        raise Exception(arg + " not an available architecture")
