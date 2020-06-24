# dataset module
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

AVAILABLE_DATASETS = ["CIFAR10", "CIFAR100"]

class CifarDataset():
    # extended: False for 10, True for 100
    def __init__(self, extended, datagen_kwargs, batch_size):
        if extended:
            (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
            self.nclasses = 100
        else:
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
            self.nclasses = 10

        # All our images are floats from [0, 1)
        X_train = X_train.astype(np.float32) / 255
        X_test = X_test.astype(np.float32) / 255
        # All our labels are categorical
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)

        datagen = ImageDataGenerator(**datagen_kwargs)
        X_test = datagen.standardize(X_test)

        self.input_shape = X_train.shape[1:]
        self.train_dataset = datagen.flow(X_train, Y_train, batch_size=batch_size)
        self.test_dataset = Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)
        self.steps_per_epoch = X_train.shape[0] // batch_size

# to-do: define ImageNet, VGGFaces
def show_available():
    print("Available datasets:", ", ".join(AVAILABLE_DATASETS))

def get_dataset(arg, datagen_kwargs, batch_size):
    if arg == "CIFAR10":
        return CifarDataset(False, datagen_kwargs, batch_size)
    if arg == "CIFAR100":
        return CifarDataset(True, datagen_kwargs, batch_size)
    elif arg not in AVAILABLE_DATASETS:
        show_available()
        raise Exception(arg + " not an available dataset")
