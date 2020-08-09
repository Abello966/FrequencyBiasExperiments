# assuming we have loose_bb_test.csv and loose_bb_train.csv on the VGGFaces2 folder.
# this can be downloaded at the VGGFaces2 institutional website
import pandas as pd
from PIL import Image
from math import floor

TARGET_SIZE = 182 #182 to be reduced to 160 via random cropping
EXPAND_SIZE = 0.2
PATH = "/misc/users/abello/VGGFaces2/" # edit this as you will

train_df = pd.read_csv(PATH + "train_df.csv", index_col="NAME_ID")
train_df_info = pd.read_csv(PATH + "loose_bb_train.csv", index_col="NAME_ID")
train_df = train_df.merge(train_df_info, on="NAME_ID")

test_df = pd.read_csv(PATH + "test_df.csv", index_col="NAME_ID")
test_df_info = pd.read_csv(PATH + "loose_bb_test.csv", index_col="NAME_ID")
test_df = test_df.merge(test_df_info, on="NAME_ID")

def preprocess_image(row):
    ex_W = row.W * (1 +  2 * EXPAND_SIZE)
    ex_H = row.H * (1 +  2 * EXPAND_SIZE)

    ex_x1 = floor(row.X - row.W * EXPAND_SIZE)
    ex_y1 = floor(row.Y - row.H * EXPAND_SIZE)

    ex_x2 = ex_x1 + ex_W
    ex_y2 = ex_y1 + ex_H

    im = Image.open(PATH + row.path)
    im = im.crop(box=(ex_x1, ex_y1, ex_x2, ex_y2))

    if im.size[0] < im.size[1]:
        scale = 182 / im.size[0]
        target = (floor(scale * im.size[0]), floor(scale * im.size[1]))
        im = im.resize(target)
        im = im.crop((0, ((target[1] - 182) // 2), 182, 182 + (target[1] - 182) // 2))

    else:
        scale = 182 / im.size[1]
        target = (floor(scale * im.size[0]), floor(scale * im.size[1]))
        im = im.resize(target)
        im = im.crop(((target[0] - 182) // 2, 0, 182 + (target[0] - 182) // 2, 182))

    im.save(PATH + row.path)

print("Preprocessing train dataset")
for _, x in train_df.iterrows():
    preprocess_image(x)

print("Preprocessing test dataset")
for _, x in test_df.iterrows():
    preprocess_image(x)
