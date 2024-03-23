import keras
import numpy as np


def normalize_dataset(dataset):
    mean = np.mean(dataset, axis=(0, 1, 2))
    std = np.std(dataset, axis=(0, 1, 2))
    return (dataset - mean) / std


    #rescale_layer = keras.layers.Rescaling(scale=1. / 255)
    #rescaled_data_set = rescale_layer(dataset)
    #layer = keras.layers.Normalization(axis=-1)
    #layer.adapt(rescaled_data_set)

    #return layer(rescaled_data_set)
