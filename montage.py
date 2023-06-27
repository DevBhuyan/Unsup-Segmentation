import os
import random
import cv2
import numpy as np


src = './ISIC 2016 for maskgen/HAIR/'
num_rows = 1  # Number of rows in the montage
num_cols = 4  # Number of columns in the montage
dst = './UNET overlay montage.png'


def create_montage(image_dir, num_rows, num_cols):
    # Get list of image filenames in the directory
    image_files = os.listdir(image_dir)

    images = []
    for image_file in image_files:
        # Read each image and resize them to the same dimensions
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (200, 200))  # You can adjust the size as needed
        images.append(image)

    # Create a blank canvas for the montage
    montage = np.zeros((200 * num_rows, 200 * num_cols, 3), dtype=np.uint8)

    # Arrange the images in the montage
    for i in range(num_rows):
        for j in range(num_cols):
            image = images[i * num_cols + j]
            montage[i * 200: (i + 1) * 200, j * 200: (j + 1) * 200] = image

    # Display the montage
    cv2.imshow("Montage", montage)
    cv2.waitKey(1)
    return montage

montage = create_montage(src, num_rows, num_cols)

cv2.imwrite(dst, montage)