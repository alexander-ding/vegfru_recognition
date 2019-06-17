from pathlib import Path
import os
from PIL import Image
import piexif

def parse(line):
    d, i = line[:-1].split(" ")
    name = d.split('/')[0]
    return (d, name, int(i))

if not data_dir.exists():
    print("data/raw does not exist. Please download the vegfru dataset and extract it at data/raw")
    exit(-1)

data_dir = Path("data") / "raw"

list_dir = data_dir / "veg200_lists"
d_train = {}
name_list = []
d_test = {}

with open(list_dir / "veg_train.txt") as f:
    for l in f.readlines():
        directory, name, i = parse(l)
        if name not in d_train.keys():
            d_train[name] = []
            name_list.append(name)
        d_train[name].append(directory)

with open(list_dir / "veg_test.txt") as f:
    for l in f.readlines():
        directory, name, i = parse(l)
        if name not in d_train.keys():
            d_train[name] = []
            name_list.append(name)
        d_train[name].append(directory)

with open(list_dir / "veg_val.txt") as f:
    for l in f.readlines():
        directory, name, i = parse(l)
        if name not in d_test.keys():
            d_test[name] = []
        d_test[name].append(directory)

list_dir = data_dir / "fru92_lists"
with open(list_dir / "fru_train.txt") as f:
    for l in f.readlines():
        directory, name, i = parse(l)
        if name not in d_train.keys():
            d_train[name] = []
            name_list.append(name)
        d_train[name].append(directory)

with open(list_dir / "fru_test.txt") as f:
    for l in f.readlines():
        directory, name, i = parse(l)
        if name not in d_train.keys():
            d_train[name] = []
            name_list.append(name)
        d_train[name].append(directory)

with open(list_dir / "fru_val.txt") as f:
    for l in f.readlines():
        directory, name, i = parse(l)
        if name not in d_test.keys():
            d_test[name] = []
        d_test[name].append(directory)

print("Gathered all the file directories")


train_dir = Path("data") / "train"
test_dir = Path("data") / "test"

if not train_dir.exists():
    print("data/train does not exist. Creating...")
    os.mkdir(str(train_dir.absolute()))

if not test_dir.exists():
    print("data/test does not exist. Creating...")
    os.mkdir(str(test_dir.absolute()))

print("Copying images...")
for name in name_list:
    base_dir = train_dir / name
    if not base_dir.exists():
        os.mkdir(base_dir)
    for j, d in enumerate(d_train[name]):
        os.rename(data_dir / d, base_dir / "{}_{}.jpg".format(name, j))

for name in name_list:
    base_dir = test_dir / name
    if not base_dir.exists():
        os.mkdir(base_dir)
    for j, d in enumerate(d_train[name]):
        os.rename(data_dir / d, base_dir / "{}_{}.jpg".format(name, j))

print("...Done")
print("Removing corrupted images...")
for name in name_list:
    for j, d in enumerate(d_train[name]):
        file_p = train_dir / name / "{}_{}.jpg".format(name, j)
        if not file_p.exists():
            continue
        with Image.open(file_p) as im:
            form = im.format
        if form != "JPEG":
            os.remove(file_p)
            continue
        im = Image.open(file_p)
        im.verify()
for name in name_list:
    for j, d in enumerate(d_test[name]):
        file_p = test_dir / name / "{}_{}.jpg".format(name, j)
        if not file_p.exists():
            continue
        with Image.open(file_p) as im:
            form = im.format
        if form != "JPEG":
            os.remove(file_p)
            continue
        im = Image.open(file_p)
        im.verify()

print("...Done")