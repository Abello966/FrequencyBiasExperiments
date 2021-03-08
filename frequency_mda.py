# generates a pickle with results of the MDA expeiment
import datasets
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as kr
import sys
import pickle as pkl

def show_use_and_exit():
    print("Use: python3 frequency_mda.py MODEL_NAME DATASET_NAME PERCENT BATCH")
    print("perform MDA on MODEL_NAME at DATASET_NAME")
    print("DATASET_NAME: available test dataset")
    print("MODEL_NAME: Keras model path")
    print("PERCENT: size of frequency disc ring ")
    print("BATCH: batch size the model was trained with")
    sys.exit()

# magic
tf.keras.backend.set_floatx('float32')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

if len(sys.argv) < 5:
    show_use_and_exit()

MODEL_NAME = sys.argv[1]
DATASET_NAME = sys.argv[2]
try:
    PERCENT = float(sys.argv[3])
except Exception as e:
    show_use_and_exit()
try:
    BATCH_SIZE = int(sys.argv[4])
except Exception as e:
    show_use_and_exit()

if len(sys.argv) >= 6:
    K = int(sys.argv[5])
else:
    K = 1

print("STARTING FREQUENCY_MDA")
print(MODEL_NAME, "on", DATASET_NAME, "with", PERCENT, " percent and batch", BATCH_SIZE)

# load dataset and get empirical distribution
if DATASET_NAME == "VGGFace2":
    dataset_kwargs = datasets.vgg_dataset_kwargs
else:
    dataset_kwargs = {}

datagen_kwargs = {
    "featurewise_center": False,  # set input mean to 0 over the dataset
    "samplewise_center": True,  # set each sample mean to 0
    "featurewise_std_normalization": False,  # divide inputs by std of the dataset
    "samplewise_std_normalization": True,  # divide each input by its std
    # set function that will be applied on each input - uncomment when normalization is not enabled
    #"preprocessing_function": lambda x: x / 255.,
}

dataset = datasets.get_test_dataset(DATASET_NAME, datagen_kwargs, BATCH_SIZE, **dataset_kwargs)
        emp_dist = utils.get_mean_energy_iterator(dataset.clean_test_datagen, dataset.input_shape)
percent_range = utils.get_percentage_masks_relevance(emp_dist, PERCENT)

# load model and calculate baseline
mod = kr.models.load_model(MODEL_NAME)
baseline_acc = utils.get_accuracy_iterator(mod, dataset.test_datagen)
print("Baseline Acc:", baseline_acc)

removed_acc = []
for i in range(len(percent_range) - 1):
    preproc = lambda Xfr: utils.remove_frequency_ring(Xfr, percent_range[i], percent_range[i + 1])
    dataset = datasets.get_test_dataset(DATASET_NAME, datagen_kwargs, BATCH_SIZE, **dataset_kwargs)

    this_mda = utils.get_accuracy_iterator(mod, dataset.test_datagen)
    print(percent_range[i], "-", percent_range[i + 1], ":", this_mda)
    removed_acc.append(this_mda)

print("MDA:")
print("Percent Range:", percent_range)
print("Perturbed Acc:", removed_acc)
output = dict()
output["percent_range"] = percent_range
output["removed_acc"] = removed_acc
output["baseline_acc"] = baseline_acc

pkl.dump(output, open(MODEL_NAME.split("/")[-1] +"_MDAtest.pkl", "wb"))
