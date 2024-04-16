import numpy as np
from sklearn.utils import shuffle
from normalization import normalize_dataset
from augmentation import augment_dataset

CLASS_SIZE = 2000


def split(dataset, class_count, split_type=1, val_set_split3=2):
    train_x_set, train_y_set, val_x_set, val_y_set, test_x_set, test_y_set = (np.array([]) for _ in range(6))
    # ranges for split 80% : 10% : 10%
    train_range = (0, int(CLASS_SIZE * 8 / 10))
    val_range = (int(CLASS_SIZE * 8 / 10), int(CLASS_SIZE * 9 / 10))
    test_range = (int(CLASS_SIZE * 9 / 10), CLASS_SIZE)
    for it in range(class_count):
        class_dataset = dataset[it * CLASS_SIZE:(it + 1) * CLASS_SIZE]
        x_set = [data_row[0] for data_row in class_dataset]
        y_set = [data_row[1] for data_row in class_dataset]
        x_set = np.array(x_set)
        y_set = np.array(y_set)
        x_set, y_set = shuffle(x_set, y_set, random_state=0)
        if it == 0:
            train_x_set = x_set[train_range[0]:train_range[1]]
            train_y_set = y_set[train_range[0]:train_range[1]]
            val_x_set = x_set[val_range[0]:val_range[1]]
            val_y_set = y_set[val_range[0]:val_range[1]]
            test_x_set = x_set[test_range[0]:test_range[1]]
            test_y_set = y_set[test_range[0]:test_range[1]]
        else:
            train_x_set = np.concatenate((train_x_set, x_set[train_range[0]:train_range[1]]))
            train_y_set = np.concatenate((train_y_set, y_set[train_range[0]:train_range[1]]))
            val_x_set = np.concatenate((val_x_set, x_set[val_range[0]:val_range[1]]))
            val_y_set = np.concatenate((val_y_set, y_set[val_range[0]:val_range[1]]))
            test_x_set = np.concatenate((test_x_set, x_set[test_range[0]:test_range[1]]))
            test_y_set = np.concatenate((test_y_set, y_set[test_range[0]:test_range[1]]))
    train_x_set, train_y_set = shuffle(train_x_set, train_y_set, random_state=0)
    val_x_set, val_y_set = shuffle(val_x_set, val_y_set, random_state=0)
    test_x_set, test_y_set = shuffle(test_x_set, test_y_set, random_state=0)
    if split_type == 2 or split_type == 3:
        train_x_set = normalize_dataset(train_x_set)
        train_x_set = augment_dataset(train_x_set)
        val_x_set = normalize_dataset(val_x_set)
        test_x_set = normalize_dataset(test_x_set)
    if split_type == 3:
        val_x_set = train_x_set[::val_set_split3]
        val_y_set = train_y_set[::val_set_split3]
    return train_x_set, train_y_set, val_x_set, val_y_set, test_x_set, test_y_set
