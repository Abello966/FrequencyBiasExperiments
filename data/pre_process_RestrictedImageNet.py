import glob
from PIL import Image
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import glob
import os
import xml.etree.ElementTree as ET

def convert_imagenet_number_name(i):
    test = np.zeros((1, 1000))
    test[0][i] = 1
    return decode_predictions(test, top=1)[0][0][0]

def preprocess_folder(division, groupclass, classname):
    if not os.path.exists("RestrictedImageNet/" + division):
        os.mkdir("RestrictedImageNet/" + division)

    if not os.path.exists("RestrictedImageNet/" + division + "/" + groupclass):
        os.mkdir("RestrictedImageNet/" + division + "/" + groupclass)

    # get list of annotations and files
    annot = glob.glob("ILSVRC2017/Annotations/CLS-LOC/" + division + "/" + classname + "/*")
    data = glob.glob("ILSVRC2017/Data/CLS-LOC/" + division + "/" + classname + "/*")

    # filter out inconsistencies
    annot_files = [x.split("/")[-1].split(".")[0] for x in annot]
    data_files = [x.split("/")[-1].split(".")[0] for x in data]
    data = [x for x in data if x.split("/")[-1].split(".")[0] in annot_files]
    annot = [x for x in annot if x.split("/")[-1].split(".")[0] in data_files]

    #since we filtered we can guarantee that annot[i] <=> data[i] if they are sorted
    data = sorted(data)
    annot = sorted(annot)

    for i in range(len(data)):
        tree = ET.parse(annot[i])
        root = tree.getroot()
        try:
            xmin = int(next(root.iter("xmin")).text)
            xmax = int(next(root.iter("xmax")).text)
            ymin = int(next(root.iter("ymin")).text)
            ymax = int(next(root.iter("ymax")).text)
        except StopIteration:
            continue

        image = Image.open(data[i])
        image = image.crop((xmin, ymin, xmax, ymax))

        name = data[i].split("/")[-1]
        image.save("RestrictedImageNet/" + division + "/" + groupclass + "/" + name)

group_class = {
    "Dog": list(range(151, 269)),
    "Cat": list(range(281, 286)),
    "Frog": list(range(30, 33)),
    "Turtle": list(range(33, 38)),
    "Bird": list(range(80, 101)),
    "Primate": list(range(365, 383)),
    "Fish": list(range(389, 398)),
    "Crab": list(range(118, 122)),
    "Insect": list(range(300, 320))
}

group_class = {key: [convert_imagenet_number_name(x) for x in value] for key, value in group_class.items()}

if not os.path.exists("RestrictedImageNet"):
    os.mkdir("RestrictedImageNet")

print("Started preprocessing: train")
for key, value in group_class.items():
    print("Starting groupclass", key)
    for classname in value:
        print("Starting classname:", classname)
        preprocess_folder("train", key, classname)
