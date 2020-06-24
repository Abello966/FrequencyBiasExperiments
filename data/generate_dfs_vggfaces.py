# assume we have identity_meta.csv, train_list and test_list all on the VGGFaces2 folder
# these can be downloaded at the VGGFaces2 institutional website
import pandas as pd

# TODO: add logic to reduce the dataset

# produce class to integer identification
df = pd.read_csv("VGGFaces2/identity_meta.csv",sep=", ")
classname_to_int = dict(zip(df.Class_ID.to_list(), df.Class_ID.index.to_list()))

train_classname = list()
train_classval = list()
train_path = list()

trainfile = open("VGGFaces2/train_list.txt")
for line in trainfile:
    line = line.rstrip()
    train_path.append("train/" + line)

    cname = line.split("/")[0]
    train_classname.append(cname)
    train_classval.append(classname_to_int[cname])

train_df = dict({"name": train_classname, "id": train_classval, "path": train_path})
train_df = pd.DataFrame(train_df)
train_df.to_csv("VGGFaces2/train_df.csv")

test_classname = list()
test_classval = list()
test_path = list()

testfile = open("VGGFaces2/test_list.txt")
for line in testfile:
    line = line.rstrip()
    test_path.append("test/" + line)

    cname = line.split("/")[0]
    test_classname.append(cname)
    test_classval.append(classname_to_int[cname])

test_df = dict({"name": test_classname, "id": test_classval, "path": test_path})
test_df = pd.DataFrame(test_df)
test_df.to_csv("VGGFaces2/test_df.csv")
