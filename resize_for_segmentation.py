import os
from PIL import Image

src = './ISIC 2016 original resized/'
target_size = (224, 224)

for folder in os.listdir(src):
    sub_dir = src+folder+'/'   # Train, Test
    print(sub_dir)
    for kind in os.listdir(sub_dir):
        kind = sub_dir+kind+'/'
        print(kind)
        for file in os.listdir(kind):
            img = Image.open(kind+file)
            if img.size != target_size:
                print('file: ', file, 'imgsize: ', img.size)
                img = img.resize(target_size)
                img.save(kind+file)
                img = Image.open(kind+file)
                if img.size != target_size:
                    raise Exception('Error in saving')
            
