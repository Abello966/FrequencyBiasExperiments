# dataset module
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

AVAILABLE_DATASETS = ["CIFAR10", "CIFAR100", "VGGFace2", "RestrictedImageNet"]

# default args
vgg_dataset_kwargs = {
    "df_path": "data/VGGFaces2/test_df.csv",
    "images_path": "data/VGGFaces2/",
    "path_col": "path",
    "class_col": "class",
    "test_samples": 2
}

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
        self.validation_steps = X_test.shape[0] // batch_size

# Assumes VGGFaceDataset has already been preprocessed
# So naturally every image has a 182x182 size

# A train dataset will not be loaded fully in memory, but in batches
# Also it will be adapted for classification

#182x182 => 160x160
def random_crop(image_in):
    assert image_in.shape[0] == image_in.shape[1], "Invalid image in shape: %d, %d" % (image_in.shape[0], image_in.shape[1])
    assert image_in.shape[0] == 182, "Invalid image in shape: %d %d" % (image_in.shape[0], image_in.shape[1])

    random_x = np.random.choice(np.arange(0, 23))
    random_y = np.random.choice(np.arange(0, 23))
    image_out = image_in[random_x:(182 - (22 - random_x)), random_y:(182 - (22 - random_y)), :]

    assert image_out.shape[0] == image_out.shape[1], "Invalid image out shape: %d, %d" % (image_out.shape[0], image_out.shape[1])
    assert image_out.shape[0] == 160, "Invalid image out shape: %d %d" % (image_out.shape[0], image_out.shape[1])

    return image_out

def crop_generator(batches, crop_length):
    while True:
        batch_x, batch_y = next(batches)
        batch_x_crop = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_x_crop[i] = random_crop(batch_x[i])

        yield (batch_x_crop, batch_y)

class VGGFaceTrainDataset():

    def __init__(self, datagen_kwargs, batch_size, df_path=None, images_path=None, path_col=None, class_col=None, test_samples=None):

        data = pd.read_csv(df_path)
        data[class_col] = data[class_col].astype("str")
        classlist = sorted(list(set(data[class_col])))

        test_dataset_df = data.groupby(class_col).apply(lambda x: x.sample(test_samples))
        test_dataset_df.index = test_dataset_df.index.droplevel()
        test_dataset_df = test_dataset_df.iloc[:(len(test_dataset_df) // batch_size) * batch_size]
        train_dataset_df = data.loc[data.index.difference(test_dataset_df.index)]

        train_dataset_df.to_csv("data/VGGFace2_train_df.csv")
        test_dataset_df.to_csv("data/VGGFace2_test_df.csv")

        datagen = ImageDataGenerator(**datagen_kwargs)

        test_generator = lambda: datagen.flow_from_dataframe(
            test_dataset_df,
            directory=images_path,
            x_col=path_col,
            y_col=class_col,
            target_size=(182, 182),
            classes=classlist,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True
        )
        self.test_dataset = Dataset.from_generator(lambda: crop_generator(test_generator(), 160),
                                                  (tf.float32, tf.float32),
                                                  (tf.TensorShape([batch_size, 160, 160, 3]), tf.TensorShape([batch_size, len(classlist)])))

        train_generator = lambda: datagen.flow_from_dataframe(
            train_dataset_df,
            directory=images_path,
            x_col=path_col,
            y_col=class_col,
            target_size=(182, 182),
            classes=classlist,
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=True
        )
        self.train_dataset = crop_generator(train_generator(), 160)

        self.input_shape = (160, 160, 3)
        self.steps_per_epoch = len(train_dataset_df) // batch_size
        self.validation_steps = len(test_dataset_df) // batch_size
        self.nclasses = len(classlist)

# A test dataset will be fully loaded and also has different logic (not classification but FR)
# class VGGFaceTestDataset

class RestrictedImageNetDataset():

    def __init__(self, datagen_kwargs, batch_size):

        datagen = ImageDataGenerator(**datagen_kwargs)

        self.train_dataset = datagen.flow_from_directory("data/RestrictedImageNet/train",
                        target_size=(160, 160), batch_size=batch_size, class_mode="categorical")
        self.test_dataset = datagen.flow_from_directory("data/RestrictedImageNet/val",
                        target_size=(160, 160), batch_size=batch_size, class_mode="categorical")


        self.input_shape = (160, 160, 3)
        self.steps_per_epoch = len(self.train_dataset)
        self.validation_steps = len(self.validation_steps)
        self.nclasses = 9



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
    if arg == "RestrictedImageNet":
        return RestrictedImageNet(datagen_kwargs, batch_size)
    elif arg not in AVAILABLE_DATASETS:
        show_available()
        raise Exception(arg + " not an available dataset")
