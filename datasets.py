# dataset module
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

AVAILABLE_DATASETS = ["CIFAR10", "CIFAR100", "VGGFace2", "RestrictedImageNet"]
AVAILABLE_TESTS = ["RestrictedImageNet"]

# default args
vgg_dataset_kwargs = {
    "df_path": "data/VGGFaces2/test_df.csv",
    "images_path": "data/VGGFaces2/",
    "path_col": "path",
    "class_col": "class",
    "test_samples": 2
}

# default datagen_kwargs
default_datagen = {
    "featurewise_center": False,  # set input mean to 0 over the dataset
    "samplewise_center": True,  # set each sample mean to 0
    "featurewise_std_normalization": False,  # divide inputs by std of the dataset
    "samplewise_std_normalization": True,  # divide each input by its std
    "zca_whitening":False,  # apply ZCA whitening
    "zca_epsilon":1e-06,  # epsilon for ZCA whitening
    "rotation_range":10,   # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    "width_shift_range":0.1,
    # randomly shift images vertically (fraction of total height)
    "height_shift_range":0.1,
    "shear_range":0.,  # set range for random shear
    "zoom_range":0.05,  # set range for random zoom
    "channel_shift_range":0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    "fill_mode":'nearest',
    "cval":0.,  # value used for fill_mode : "constant"
    "horizontal_flip":True,  # randomly flip images
    "vertical_flip":False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    "rescale":None,
    # set function that will be applied on each input
    "preprocessing_function":None,
    # image data format, either "channels_first" or "channels_last"
    "data_format":"channels_last",
}

class CifarDataset():
    # extended: False for 10, True for 100
    def __init__(self, extended, datagen_kwargs, batch_size, val_split=0.1):
        if extended:
            (X_train, Y_train), (_, _) = cifar100.load_data()
            self.nclasses = 100
        else:
            (X_train, Y_train), (_, _) = cifar10.load_data()
            self.nclasses = 10

        # Divide train/val
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=val_split, stratify=Y_train, random_state=2)

        # All our labels are categorical and all our images are floats
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)

        datagen = ImageDataGenerator(**datagen_kwargs)
        X_train = datagen.standardize(X_train.astype(np.float32))
        X_test = datagen.standardize(X_test.astype(np.float32))

        self.input_shape = X_train.shape[1:]
        self.train_dataset = Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size).repeat()
        self.test_dataset = Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)
        self.steps_per_epoch = X_train.shape[0] // batch_size
        self.validation_steps = X_test.shape[0] // batch_size


class CifarTestDataset():
    # extended: False for 10, True for 100
    def __init__(self, extended, datagen_kwargs, batch_size):
        if extended:
            (_, _), (X_test, Y_test) = cifar100.load_data()
            self.nclasses = 100
        else:
            (_, _), (X_test, Y_test) = cifar10.load_data()
            self.nclasses = 10

        # All our images are floats from [0, 1)
        X_test = X_test.astype(np.float32) / 255
        # All our labels are categorical
        Y_test = to_categorical(Y_test)

        datagen = ImageDataGenerator(**datagen_kwargs)
        X_test = datagen.standardize(X_test)

        self.test_datagen = Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)
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

def train_test_split_VGG(df_path, class_col, random_seed=0):
    data = pd.read_csv(df_path)

    test_dataset_df = data.groupby(class_col).apply(lambda x: x.sample(frac=test_frac, random_state=random_seed))
    test_dataset_df.index = test_dataset_df.index.droplevel()
    train_dataset_df = data.loc[data.index.difference(test_dataset_df.index)]

    train_dataset_df.to_csv("data/" + str(datetime.date.today()) + "_VGGFace2_train_df.csv")
    test_dataset_df.to_csv("data/" + str(datetime.date.today()) + "_VGGFace2_test_df.csv")

class VGGFaceTrainDataset():

    def __init__(self, datagen_kwargs, batch_size, df_path=None, images_path=None, path_col=None, class_col=None, test_frac=None):
        train_dataset_df = pd.read_csv(df_path)
        train_dataset_df[class_col] = train_dataset_df[class_col].astype("str")
        classlist = sorted(list(set(train_dataset_df[class_col])))

        datagen = ImageDataGenerator(**datagen_kwargs)

        master_generator = lambda split: datagen.flow_from_dataframe(
            train_dataset_df,
            directory=images_path,
            x_col=path_col,
            y_col=class_col,
            target_size=(160, 160),
            classes=classlist,
            batch_size=batch_size,
            class_mode="categorical",
            subset=split,
            shuffle=True
        )
        )
        self.train_dataset = master_generator("training")
        self.input_shape = (160, 160, 3)
        self.steps_per_epoch = len(train_dataset_df) // batch_size
        self.nclasses = len(classlist)

# A test dataset will be fully loaded and also has different logic (not classification but FR)
# class VGGFaceTestDataset

class RestrictedImageNetDataset():

    def __init__(self, datagen_kwargs, batch_size, validation_split=0.1):

        datagen = ImageDataGenerator(**datagen_kwargs, validation_split=validation_split)

        self.train_dataset = datagen.flow_from_directory("data/RestrictedImageNet/train",
                        target_size=(160, 160), batch_size=batch_size, class_mode="categorical", subset="training")
        self.test_dataset = datagen.flow_from_directory("data/RestrictedImageNet/train",
                        target_size=(160, 160), batch_size=batch_size, class_mode="categorical", subset="validation")


        self.input_shape = (160, 160, 3)
        self.steps_per_epoch = len(self.train_dataset)
        self.validation_steps = len(self.test_dataset)
        self.nclasses = 9

class RestrictedImageNetDatasetTest():
    def __init__(self, datagen_kwargs, batch_size):
        datagen = ImageDataGenerator(**datagen_kwargs)
        self.test_datagen = datagen.flow_from_directory("data/RestrictedImageNet/val",
                        target_size=(160,160), batch_size=batch_size, class_mode="categorical")


def show_available(av_list):
    print("Available datasets:", ", ".join(av_list))

def get_dataset(arg, datagen_kwargs, batch_size, **kwargs):
    if arg == "CIFAR10":
        return CifarDataset(False, datagen_kwargs, batch_size)
    if arg == "CIFAR100":
        return CifarDataset(True, datagen_kwargs, batch_size)
    if arg == "VGGFace2":
        return VGGFaceTrainDataset(datagen_kwargs, batch_size, **kwargs)
    if arg == "RestrictedImageNet":
        return RestrictedImageNetDataset(datagen_kwargs, batch_size)
    elif arg not in AVAILABLE_DATASETS:
        show_available(AVAILABLE_DATASETS)
        raise Exception(arg + " not an available dataset")

def get_test_dataset(arg, datagen_kwargs, batch_size):
    if arg == "RestrictedImageNet":
        return RestrictedImageNetDatasetTest(datagen_kwargs, batch_size)
    else:
        show_available(AVAILABLE_TESTS)
        raise Exception(arg + " not an available test dataset")
