# dataset module
import numpy as np
import pandas as pd
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

AVAILABLE_DATASETS = ["CIFAR10", "CIFAR100", "VGGFace2"]

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
        self.test_dataset = Dataset.from_tensor_slices((X_test, Y_test)).batch(X_test.shape[0])
        self.steps_per_epoch = X_train.shape[0] // batch_size

# Assumes VGGFaceDataset has already been preprocessed
# So naturally every image has a 182x182 size

# A train dataset will not be loaded fully in memory, but in batches
# Also it will be adapted for classification
class VGGFaceTrainDataset():

    #182x182 => 160x160
    def random_crop(image_in):
        random_x = np.random.choice(np.arange(0, 23))
        random_y = np.random.choice(np.arange(0, 23))
        return image_in[random_x:(182 - (22 - random_x)), random_y:(182 - (22 - random_y)), :]

    # TODO pandas logic
    def __init__(self, datagen_kwargs, batch_size, df_path=None, images_path=None, path_col=None, class_col=None, test_samples=None):

        data = pd.read_csv(df_path)
        data[class_col] = data[class_col].astype("str")
        classlist = sorted(list(set(data[class_col])))

        test_dataset = data.groupby(class_col).apply(lambda x: x.sample(test_samples))
        test_dataset.index = test_dataset.index.droplevel()
        train_dataset = data.loc[data.index.difference(test_dataset.index)]

        datagen = ImageDataGenerator(**{**datagen_kwargs, **{"preprocessing_function": self.random_crop}})

        self.test_dataset = datagen.flow_from_dataframe(
            test_dataset,
            directory=images_path,
            x_col=path_col,
            y_col=class_col,
            target_size=(160, 160),
            classes=classlist,
            class_mode="categorical",
            batch_size=batch_size
        )

        self.train_dataset = datagen.flow_from_dataframe(
            train_dataset,
            directory=images_path,
            x_col=path_col,
            y_col=class_col,
            target_size=(160, 160),
            classes=classlist,
            class_mode="categorical",
            batch_size=batch_size
        )

        self.input_shape = (160, 160, 3)
        self.nclasses = len(classlist)

# A test dataset will be fully loaded and also has different logic (not classification but FR)
# class VGGFaceTestDataset


# to-do: define ImageNet, VGGFaces
def show_available():
    print("Available datasets:", ", ".join(AVAILABLE_DATASETS))

def get_dataset(arg, datagen_kwargs, batch_size, **kwargs):
    if arg == "CIFAR10":
        return CifarDataset(False, datagen_kwargs, batch_size)
    if arg == "CIFAR100":
        return CifarDataset(True, datagen_kwargs, batch_size)
    if arg == "VGGFace2":
        return VGGFaceTrainDataset(datagen_kwargs, batch_size, **kwargs)
    elif arg not in AVAILABLE_DATASETS:
        show_available()
        raise Exception(arg + " not an available dataset")
