# generates a pickle with results of the MDA expeiment
import datasets
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as kr
import sys

def show_use_and_exit():
    print("Use: python3 frequency_mda.py DATASET_NAME MODEL_NAME PERCENT BATCH")
    print("perform MDA with test dataset DATASET_NAME on given MODEL_NAME")
    print("DATASET_NAME: available test dataset")
    print("MODEL_NAME: Keras model path")
    print("PERCENT: ")
    sys.exit()

# magic
tf.keras.backend.set_floatx('float32')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

if len(sys.argv) != 5:
    show_use_and_exit()

DATASET_NAME = sys.argv[1]
MODEL_NAME = sys.argv[2]
try:
    PERCENT = float(sys.argv[3])
except Exception as e:
    show_use_and_exit()
try:
    BATCH_SIZE = int(sys.argv[4])
except Exception as e:
    show_use_and_exit()

print("STARTING FREQUENCY_MDA")
print(MODEL_NAME, "on", DATASET_NAME, "with", PERCENT, " percent and batch", BATCH_SIZE)

# load dataset and get empirical distribution
dataset = datasets.get_test_dataset(DATASET_NAME, datasets.default_datagen, BATCH_SIZE)
emp_dist = utils.get_mean_energy_iterator(dataset.test_datagen)
percent_range = utils.get_percentage_masks_relevance(emp_dist, PERCENT)

# load model and calculate baseline
mod = kr.models.load_model("model_weights/" + MODEL_NAME)
baseline_acc = utils.get_accuracy_iterator(mod, dataset.test_datagen)
print("Baseline Acc:", baseline_acc)

removed_acc = []
for i in range(len(percent_range) - 1):
    preproc = lambda Xfr: utils.remove_frequency_ring_dataset(Xfr, percent_range[i], percent_range[i + 1])
    removed_acc.append(utils.get_accuracy_iterator(mod, dataset.test_datagen, preproc=preproc))

print("MDA:")
print("Percent Range:", percent_range)
print("Perturbed Acc:", removed_acc)
output = dict()
output["percent_range"] = percent_range
output["removed_acc"] = removed_acc

pkl.dump(output, open(MODEL_NAME +"_MDA.pkl", "wb"))
