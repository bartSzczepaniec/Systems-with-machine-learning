import imgaug.augmenters as iaa
import random
import numpy as np

def augment_dataset(dataset, dataset_labels):
    augmented_dataset = []
    augmented_dataset_labels = []
    seq = iaa.Sequential()

    for index, img in enumerate(dataset):
        img_arr = img

        rand = random.uniform(0, 1)
        if rand < 0.25:
            seq = iaa.Sequential([iaa.imgcorruptlike.GaussianNoise(severity=1)])
        elif rand < 0.5:
            seq = iaa.Sequential([iaa.Fliplr(1.0)])
        elif rand < 0.75:
            seq = iaa.Sequential([iaa.Flipud(1.0)])
        else:
            seq = iaa.Sequential([iaa.Rotate(90)])

        augmented_dataset.append(seq(image=img_arr))
        augmented_dataset_labels.append(dataset_labels[index])
    
    augmented_dataset = np.array(augmented_dataset)
    augmented_dataset_labels = np.array(augmented_dataset_labels)
    return np.concatenate((dataset, augmented_dataset)), np.concatenate((dataset_labels, augmented_dataset_labels))
