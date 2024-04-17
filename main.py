import os
import shutil

from PIL import Image as PILImage
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from data_splits import split
from resize_images import resize_images
from normalization import normalize_dataset
from sklearn.utils import shuffle
from utils import load_dataset

DATASET_PATH = "./images"
RESIZED_DATA_PATH = "./resized_images"
class_to_number = {
    "Burger": 0,
    "Donut": 1,
    "Hot dog": 2,
    "Pizza": 3,
    "Sandwich": 4
}

number_to_class = {
    0: "Burger",
    1: "Donut",
    2: "Hot dog",
    3: "Pizza",
    4: "Sandwich"
}


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def change_file_names(dir, class_name):
    folder = os.path.join(dir, class_name)
    idx = 1
    for file in os.listdir(folder):
        file_dir = os.path.join(folder, file)
        new_name = f"{class_name}_{idx}.jpeg"
        os.rename(file_dir, os.path.join(folder, new_name))
        idx += 1



images_path = "D:\SZuM\images\Fast Food Data"
#copytree(images_path + "\Training Data\Sandwich", "./images/Sandwich")
#copytree(images_path + "\Training Data\Burger", "./images/Burger")
#copytree(images_path + "\Training Data\Donut", "./images/Donut")
#copytree(images_path + "\Training Data\Hot Dog", "./images/Hot Dog")
#copytree(images_path + "\Training Data\Pizza", "./images/Pizza")

#copytree(images_path + "\Validation Data\Sandwich", "./images/Sandwich")
#copytree(images_path + "\Validation Data\Burger", "./images/Burger")
#copytree(images_path + "\Validation Data\Donut", "./images/Donut")
#copytree(images_path + "\Validation Data\Hot Dog", "./images/Hot Dog")
#copytree(images_path + "\Validation Data\Pizza", "./images/Pizza")

#for class_name in ("Burger", "Donut", "Hot Dog", "Pizza", "Sandwich"):
#    change_file_names(DATASET_PATH, class_name)

class_names = sorted(os.listdir(DATASET_PATH))

#resize_images(DATASET_PATH, class_names, new_size=(128, 128))

# Splits

dataset = load_dataset(RESIZED_DATA_PATH, class_names)
#train_x_set, train_y_set, val_x_set, val_y_set, test_x_set, test_y_set = split(dataset, 5, split_type=1)
#train_x_set, train_y_set, val_x_set, val_y_set, test_x_set, test_y_set = split(dataset, 5, split_type=2)
train_x_set, train_y_set, val_x_set, val_y_set, test_x_set, test_y_set = split(dataset, 5, split_type=3)

#np.savez('data_split.npz', train_x=train_x_set, train_y=train_y_set, val_x=val_x_set, val_y=val_y_set, test_x=test_x_set, test_y=test_y_set)


def create_tfrecord(x_set, y_set, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        it = 0
        for x, y in zip(x_set, y_set):
            feature = {
                'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten())),
                'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            print(it)
            it += 1
print("SAVING TO TFRECORD FILES")
# Save your data splits to TFRecord files
start = time.time()
create_tfrecord(train_x_set, train_y_set, 'train.tfrecord')
create_tfrecord(val_x_set, val_y_set, 'val.tfrecord')
create_tfrecord(test_x_set, test_y_set, 'test.tfrecord')
end = time.time()
print("Saving to tfrecord file time: " + str(end - start))

print(train_x_set)
print(len(val_x_set))
print(len(test_x_set))
print(len(train_x_set))

#PILImage.fromarray(x_set[2], 'RGB').show()
# Splits
#x_set, y_set = shuffle(x_set, y_set)
# Split 1





#x_set = normalize_dataset(x_set)

#print(x_set)
#x_set = normalize_dataset(x_set)
#ft = x_set#[:].numpy()
#t = x_set[2]#.numpy()
#print(number_to_class[y_set[2]])
#PILImage.fromarray(((t-ft.min())/(ft.max()-ft.min())*255).astype(np.uint8), 'RGB').show()
#print(x_set)
#print("Max - > min")
#print(np.max(x_set))
#print(np.min(x_set))