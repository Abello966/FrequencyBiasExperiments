# generate examples of degraded pictures for each dataset
import datasets
import utils
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle as pkl

def show_use_and_exit():
    print("USE: frequency_examples.py DATASET_NAME PERCENT NUM_EXAMPLES" )
    sys.exit()

if len(sys.argv) < 3:
    show_use_and_exit()

DATASET_NAME = sys.argv[1]
try:
    PERCENT = float(sys.argv[2])
except Exception as e:
    show_use_and_exit()
try:
    NUM_EXAMPLES = int(sys.argv[3])
except Exception as e:
    show_use_and_exit()

vgg_dataset_kwargs = {
    "df_path": "data/2020-09-20_VGGFace2_test_df.csv",
    "images_path": "data/VGGFaces2/",
    "path_col": "path",
    "class_col": "class",
}

seed = 1

# load dataset and get empirical distribution
if DATASET_NAME == "VGGFace2":
    dataset_kwargs = vgg_dataset_kwargs
else:
    dataset_kwargs = {}

dataset = datasets.get_test_dataset(DATASET_NAME, {}, NUM_EXAMPLES, seed=seed, **dataset_kwargs)
emp_dist = utils.get_mean_energy_iterator(dataset.clean_test_datagen, dataset.input_shape)
percent_range = utils.get_percentage_masks_relevance(emp_dist, PERCENT)
baseline_sample, _ = next(dataset.test_datagen)
Xsamples = []
for i in range(len(percent_range) - 1):
    preproc = lambda Xfr: utils.remove_frequency_ring(Xfr, percent_range[i], percent_range[i + 1])
    dataset = datasets.get_test_dataset(DATASET_NAME, {"preprocessing_function": preproc},
                                        NUM_EXAMPLES, **dataset_kwargs)
    Xsample, _ = next(dataset.test_datagen)
    Xsamples.append(Xsample)


print("Examples:")
print("Percent Range:", percent_range)
output = dict()
output["baseline_sample"] = baseline_sample
output["percent_range"] = percent_range
output["Xsamples"] = Xsamples
pkl.dump(output, open(DATASET_NAME + "_EXAMPLES.pkl", "wb"))
