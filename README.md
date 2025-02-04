# Frequency Bias Experiments
This repository allows for both training and evaluating frequency bias in CNNs.
Currently supported datasets are:

- VGGFace2
- ILSVRC2017/RestrictedImageNet (as defined by Ilyas et al 2019)
- SVHN
- CIFAR10

This README walks the reader through the process of reproducing the experiments of
A Systematic Study of Frequency Bias in CNN (publication pending), from pre-processing
to graphics generation.

## Dataset Pre-processing
CIFAR10 and SVHN datasets can be used as they are provided by the Keras and Tensorflow
libraries respectively.

The two other datasets require some preprocessing. We include the "data" folder with
the scripts used to preprocess the data, but not the data itself.

For ILSVRC2017/ImageNet the pre\_process\_RestrictedImageNet crops images and groups them
in the 9 super-classes used in RestrictedImageNet, organizing folders in a way that's
convenient for the Keras ImageDataGenerator's "flow\_from\_directory" method

For VGGFace2, "pre\_process\_vggfaces.py" will crop and center images, while "generate\_dfs\_vggfaces.py"
will consolidate the disperse information in various csvs provided in the VGGFace2 dataset into
a single pandas dataframe/csv.

Since VGGFace2 is not a classification dataset, we need to construct train/test splits
of our own. This is done through the datasets.py "train\_test\_split\_VGG" method, which receives
the aforementioned VGGFace2 dataframe's path and splits it into a train and a test one.

## Model Training
Model training is done through the "train\_model\_*" family of python scripts. Originally it
was intended for it to be a single script, but particularities of each dataset made it
easier to separate each case in a different file. As a result, the code is not exactly DRY
as it is now.

Models' architectures are imported from "arch.py", which serves as an interface for
Keras' default models aswell as our own implementation of some architectures.

The training script saves model weights into the model\_weights folder and generates a
pickle of the history dictionary generated by Keras during training recording relevant
training statistics.

## Model Frequency Bias Estimation
frequency\_mda.py estimate the frequency bias of a given trained model on a given dataset.
He estimates the test dataset energy distribution and the frequency bands according to
our method (see paper). After that he calculates the baseline performance of the model on
the test dataset, and the performance on the distorted versions of the dataset. The script
generates a pickle containing a dictionary of the results.

## Dataset Example Generation
frequency\_examples.py cherrypicks random examples of a dataset, as well as produces the
distorted versions of each example according to our method (see paper). The script
generates a pickle as a result.

## Results Compilation
Results compilation and graphics are done through the Results.ipynb jupyter notebook.
