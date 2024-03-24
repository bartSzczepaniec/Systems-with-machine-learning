import numpy as np
from scipy.ndimage import rotate
import imgaug.augmenters as iaa


def translate(img, shift=10, direction='right', bg_patch=(5, 5)):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    if direction == 'right':
        img[:, shift:] = img[:, :-shift]
        img[:, :shift] = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0, 1))
    if direction == 'left':
        img[:, :-shift] = img[:, shift:]
        img[:, -shift:] = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0, 1))
    if direction == 'down':
        img[shift:, :] = img[:-shift, :]
        img[:shift, :] = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0, 1))
    if direction == 'up':
        img[:-shift, :] = img[shift:, :]
        img[-shift:, :] = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0, 1))
    return img


def rotate_img(img, angle, bg_patch=(5, 5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0, 1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


def flip_img(img, direction='horizontal'):
    assert direction in ['horizontal', 'vertical'], 'Direction should be horizontal or vertical'
    img = img.copy()
    if direction == 'horizontal':
        img = np.fliplr(img)
    elif direction == 'vertical':
        img = np.flipud(img)
    return img


def apply_random_transformation(img):
    transformations = [translate, rotate_img, flip_img]
    chosen_transformation = np.random.choice(transformations)
    if chosen_transformation == rotate_img:
        angle = np.random.randint(0, 360)
        augmented_img = chosen_transformation(img, angle)
    else:
        augmented_img = chosen_transformation(img)
    return augmented_img


number_to_class = {
    0: "Burger",
    1: "Donut",
    2: "Hot Dog",
    3: "Pizza",
    4: "Sandwich"
}


def augment_dataset(dataset):
    augmented_dataset = []
    seq = iaa.Sequential([
        iaa.ChangeColorTemperature((1000, 10000)),
        iaa.imgcorruptlike.GaussianNoise(severity=4)
    ])

    for img in dataset:
        img_arr = img[0]
        img_class_number = img[1]

        augmented_dataset.append(img)

        augmented_dataset.append([flip_img(img_arr, 'vertical'), img_class_number])
        augmented_dataset.append([rotate_img(img_arr, 45), img_class_number])
        augmented_dataset.append([rotate_img(img_arr, 135), img_class_number])
        augmented_dataset.append([rotate_img(img_arr, 225), img_class_number])
        augmented_dataset.append([rotate_img(img_arr, 315), img_class_number])

        flipped_img_arr = flip_img(img_arr)
        augmented_dataset.append([flip_img(flipped_img_arr, 'vertical'), img_class_number])
        augmented_dataset.append([rotate_img(flipped_img_arr, 45), img_class_number])
        augmented_dataset.append([rotate_img(flipped_img_arr, 135), img_class_number])
        augmented_dataset.append([rotate_img(flipped_img_arr, 225), img_class_number])
        augmented_dataset.append([rotate_img(flipped_img_arr, 315), img_class_number])

        augmented_dataset.append([seq(image=img_arr), img_class_number])