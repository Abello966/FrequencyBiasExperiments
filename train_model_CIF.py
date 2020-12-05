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
    # randomly shift images horizontally (fraction of total width)
    "width_shift_range":0.1,
    # randomly shift images vertically (fraction of total height)
    "height_shift_range":0.1,
    "horizontal_flip":True,  # randomly flip images
    "samplewise_center": True,  # set each sample mean to 0
    "samplewise_std_normalization": True,  # divide each input by its std
}

# TODO: this should be put on a .env perhaps
# VGGFace
vgg_dataset_kwargs = {
    "df_path": "/misc/users/abello/VGGFaces2/train_df.csv",
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

if todays_ds == "VGGFace2":
    dataset_kwarg = vgg_dataset_kwargs
else:
    dataset_kwarg = empty_kwargs

EPOCHS = 100
batch_size = 128
NAME = str(datetime.date.today()) + "_" + todays_ds + "NORMALIZED_" + str(batch_size) + todays_mod

dataset = datasets.get_dataset(todays_ds, datagen_kwargs, batch_size, val_split=0.1, **dataset_kwarg)
#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():

def lr_scheduler(epoch, lr):
    lr = 1e-2
    if epoch > 80:
        lr = lr / 10
    return lr

def no_scheduler(epoch, lr):
    return lr
            
model = arch.get_arch(todays_mod, dataset.input_shape, dataset.nclasses, **model_kwargs)

LRS = kr.callbacks.LearningRateScheduler(no_scheduler, verbose=1)

MCC = kr.callbacks.ModelCheckpoint(
    filepath="model_weights/" + NAME,
    monitor="val_accuracy",
    save_best_only=True)
    
opt = kr.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
#opt = kr.optimizers.Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

hist = model.fit(dataset.train_dataset, epochs=EPOCHS, steps_per_epoch=dataset.steps_per_epoch,
                validation_steps=dataset.validation_steps, validation_data=dataset.test_dataset,
        callbacks=[MCC, LRS], use_multiprocessing=False, workers=1, max_queue_size=128)
pkl.dump(hist.history, open(NAME + ".pkl", "wb"))
