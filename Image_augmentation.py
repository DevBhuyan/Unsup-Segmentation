from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os
import numpy as np

cwd = '/mnt/C654078D54078003/Curious Dev B/PROJECT STAGE - II/ISIC 2016 for segmentation (augmented)/Train/IMAGES/'
print(cwd)

ls = os.listdir(cwd)
l = len(ls)
rem = 1500 - l

for i in range(l):
    if rem and ls[i][:3] != 'AUG':
        file = ls[i]
        im = io.imread(cwd+file)
        im = np.expand_dims(im, axis=0)
        c = 0
        if i%4 == 0:
            x = 0.4
            y = z = w = 0
        elif i%4 == 1:
            y = 0.4
            x = z = w = 0
        elif i%4 == 2:
            z = 0.4
            x = y = w = 0
        else:
            w = 0.4
            x = y = z = 0
        datagen = ImageDataGenerator(
            width_shift_range=x,  
            height_shift_range=y,    
            shear_range=z,        
            zoom_range=w)
        for batch in datagen.flow(im,
                                  batch_size=1,
                                  save_to_dir=cwd,
                                  save_prefix='AUG',
                                  save_format='jpg'):
            c += 1
            if c:
                break
    else:
        break
    rem -= 1
    
    