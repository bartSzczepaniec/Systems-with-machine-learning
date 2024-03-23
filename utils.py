import os
import numpy as np
from PIL import Image as PILImage

class_to_number = {
    "Burger": 0,
    "Donut": 1,
    "Hot dog": 2,
    "Pizza": 3,
    "Sandwich": 4
}


def load_dataset(path_to_dataset, class_names):
    print(f"Loading whole dataset from directory: {path_to_dataset}")

    dataset = []
    for class_name in class_names:
        for img_name in sorted(os.listdir(os.path.join(path_to_dataset, class_name))):
            img = np.array(PILImage.open(os.path.join(path_to_dataset, class_name, img_name)))
            dataset.append([img, class_to_number[class_name]])

    return dataset
