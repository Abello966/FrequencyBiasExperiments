import pickle as pkl
import sys
import tensorflow as tf
import tensorflow.keras as kr
import datasets
import datetime
import arch
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# magic
tf.keras.backend.set_floatx('float32')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def show_use_and_exit():
    print("train_model: train an ARCHITECTURE on a specific DATASET")
    print("USE: python3 train_model.py ARCHITECTURE DATASET")
    arch.show_available()
    datasets.show_available(datasets.AVAILABLE_DATASETS)
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
    "preprocessing_function": lambda x: x / 255.,
    # image data format, either "channels_first" or "channels_last"
    "data_format":"channels_last",
}

# TODO: this should be put on a .env perhaps
# VGGFace
vgg_dataset_kwargs = {
    "df_path": "data/2020-09-20_VGGFace2_train_df.csv",
    "images_path": "data/VGGFaces2/",
    "path_col": "path",
    "class_col": "class",
    "test_frac": 0.05,
}

empty_kwargs = {}

model_kwargs = {
    #"Normalization": "BatchNormalization"
}

if len(sys.argv) != 3:
    show_use_and_exit()

todays_mod = sys.argv[1]
todays_ds = sys.argv[2]

EPOCHS = 10
batch_size = 64
NAME = str(datetime.date.today()) + todays_ds + str(batch_size) + todays_mod

dataset = datasets.get_dataset("VGGFace2", datagen_kwargs, batch_size, **vgg_dataset_kwargs)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    def lr_scheduler(epoch, lr):
        if epoch < 5:
            return lr
        elif epoch % 5 == 0:
            return lr / 2
        else:
            return lr

    def no_scheduler(epoch, lr):
        return lr
            
    model = arch.get_arch(todays_mod, dataset.input_shape, dataset.nclasses, **model_kwargs)

    LRS = kr.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    MCC = kr.callbacks.ModelCheckpoint(
        filepath="model_weights/" + NAME + "_{epoch}",
        monitor="val_loss",
        save_best_only=False)
    
    opt = kr.optimizers.SGD(lr=1e-2, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


hist = model.fit(dataset.train_dataset, epochs=EPOCHS,
        steps_per_epoch=dataset.steps_per_epoch, callbacks=[MCC, LRS],
        use_multiprocessing=True, workers=32, max_queue_size=128)
pkl.dump(hist.history, open(NAME + ".pkl", "wb"))
