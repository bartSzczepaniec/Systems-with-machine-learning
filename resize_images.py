import os
from PIL import Image as PILImage


def resize_images(path_to_dataset, class_names, new_size=(128, 128)):
    path_to_resized_dataset = "./resized_images"
    for class_name in class_names:
        for img_name in sorted(os.listdir(os.path.join(path_to_dataset, class_name))):
            img = PILImage.open(os.path.join(path_to_dataset, class_name, img_name))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.resize(new_size).save(os.path.join(path_to_resized_dataset, class_name, img_name))
