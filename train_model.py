import sys
import tensorflow as tf
import tensorflow.keras as kr
import datasets
import datetime
import arch
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def show_use_and_exit():
    print("train_model: train an ARCHITECTURE on a specific DATASET")
    print("USE: python3 train_model.py ARCHITECTURE DATASET")
    arch.show_available()
    datasets.show_available()
    sys.exit()

datagen_kwargs = {
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

# VGGFace
vgg_dataset_kwargs = {
    "df_path": "data/VGGFaces2/test_df.csv",
    "images_path": "data/VGGFaces2/",
    "path_col": "path",
    "class_col": "class",
    "test_samples": 2
}

empty_kwargs = {}

model_kwargs = {
    #"Normalization": "BatchNormalization"
}

if len(sys.argv) != 3:
    show_use_and_exit()

# magic
tf.keras.backend.set_floatx('float32')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


todays_mod = sys.argv[1]
todays_ds = sys.argv[2]

if todays_ds == "VGGFace2":
    dataset_kwarg = vgg_dataset_kwargs
else:
    dataset_kwarg = empty_kwargs

NAME = str(datetime.date.today()) + "_" + todays_ds + "_" + todays_mod + ""
EPOCHS = 200
batch_size = 4

dataset = datasets.get_dataset(todays_ds, datagen_kwargs, batch_size, **dataset_kwarg)
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = arch.get_arch(todays_mod, dataset.input_shape, dataset.nclasses, **model_kwargs)

    MCC = kr.callbacks.ModelCheckpoint(
        filepath="model_weights/" + NAME,
        monitor="val_loss",
        save_best_only=True)
    opt = kr.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(dataset.train_dataset, validation_data=dataset.test_dataset, epochs=EPOCHS,
        steps_per_epoch=dataset.steps_per_epoch, validation_steps=dataset.validation_steps, callbacks=[MCC],
        use_multiprocessing=True, workers=4, max_queue_size=12)
