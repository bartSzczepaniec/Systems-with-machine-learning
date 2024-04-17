import imgaug.augmenters as iaa
import random
import numpy as np

def augment_dataset(dataset):
    augmented_dataset = []
    seq = iaa.Sequential()

    for img in dataset:
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
    
    augmented_dataset = np.array(augmented_dataset)
    return np.concatenate((dataset, augmented_dataset))
