# assume we have identity_meta.csv, train_list and test_list all on the VGGFaces2 folder
# these can be downloaded at the VGGFaces2 institutional website
import pandas as pd

PATH = "VGGFaces2/"

# produce class to integer identification
df = pd.read_csv(PATH + identity_meta.csv",sep=", ")
classname_to_int = dict(zip(df.Class_ID.to_list(), df.Class_ID.index.to_list()))

train_classname = list()
train_nameid = list()
train_classval = list()
train_path = list()

trainfile = open(PATH + "train_list.txt")
for line in trainfile:
    line = line.rstrip()
    train_path.append("train/" + line)
    train_nameid.append(line[0:-4])

    cname = line.split("/")[0]
    train_classname.append(cname)
    train_classval.append(classname_to_int[cname])

train_df = dict({"NAME_ID": train_nameid, "name": train_classname, "class": train_classval, "path": train_path})
train_df = pd.DataFrame(train_df)
train_df = train_df.set_index("NAME_ID")
train_df.to_csv(PATH + "train_df.csv")

test_classname = list()
test_nameid = list()
test_classval = list()
test_path = list()

testfile = open(PATH + "test_list.txt")
for line in testfile:
    line = line.rstrip()
    test_path.append("test/" + line)
    test_nameid.append(line[0:-4])

    cname = line.split("/")[0]
    test_classname.append(cname)
    test_classval.append(classname_to_int[cname])

test_df = dict({"NAME_ID": test_nameid, "name": test_classname, "class": test_classval, "path": test_path})
test_df = pd.DataFrame(test_df)
test_df = test_df.set_index("NAME_ID")
test_df.to_csv(PATH + "test_df.csv")
