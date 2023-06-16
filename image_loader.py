import os
import numpy as np
from PIL import Image


def is_image_file(file_path):
    # Check if the file has a supported image extension
    extensions = ('.jpg', '.jpeg', '.png', '.gif')
    return any([file_path.lower().endswith(ext) for ext in extensions])


def import_images_from_directory(directory):
    images = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the file is an image
        if is_image_file(file_path):
            # Import the image and add it to the list
            image = import_image(file_path)
            images.append(image)

    return np.array(images)


def import_image(file_path):
    # Open the image file
    image = Image.open(file_path)

    # Convert the image to RGB mode if it's not already in that mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert image to numpy array
    image = np.array(image)

    return image


if __name__ == "__main__":
    directory_path = 'EXEMPLAR'
    images = import_images_from_directory(directory_path)
    print(len(images))
    np.save(f"loaded_imgs/loaded_{directory_path}.npy", images)
    