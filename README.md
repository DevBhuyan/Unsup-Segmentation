# Unsup-Segmentation

This repository contains code that was a part of the project "SEGMENTATION AND CLASSIFICATION OF SKIN LESION IMAGES USING DEEP LEARNING BASED TECHNIQUES", published as part of my [final year project](https://drive.google.com/file/d/176D-SWsVus7_6dfPSnIhaun2YDx_5Nv_/view?usp=sharing). This was an experiment to create an unsupervised segmentation pipeline that can create segmentation masks of any given skin lesion image from the ISIC datasets. Although this was predominantly tested on the 2016 and 2017 datasets, this can be extended to latter datasets as well. These are the results tested on the 2016 dataset.

![image](https://github.com/DevBhuyan/Unsup-Segmentation/assets/55915667/7f81729b-cd6f-45c8-a387-ce8cd9846ea4)

The complete pipeline will require you to run the "mask_generator.py" file first, which should save the unfiltered mask images into an 'UNFILTERED/' directory inside your 'MASKS/' directory. After that, you can run either of "color_clean.py" or "texture.py" (first one being a simpler cleaning approach while the second one being a slightly more complicated but accurate approach), this will create the final masks for you. At any point of time you can run the "metrics.py" file to check the performance of the model. You can run all of these algorithms sequentially or in parellel. Graph execution won't be an issue as they read each image and process it in real time. For example, if you have been running `mask_generator` for a few minutes and have a few images in the `UNFILTERED/` directory, you can run `texture` which will start working on the unfiltered images in real-time. `tqdm` will consider the existing number of files and start accordingly. Simiarly, you can run `metrics` at any point to check the performance on the masks generated so far.

You can download the original ISIC 2016 dataset from https://challenge.isic-archive.com/data/. Or you can try this on your own dataset. To run `metrics`, you'll need to have reference masks against which the generated masks will be compared.

You can play with "permute_channels.py" to get an idea of how the different channels interact when combined. 

"mask_overlay.py" will help you draw the mask outlines on the original images so that you can have a better idea of how the algorithm is performing.

Feel free to play around with the code. Also make sure to modify the directory locations as required.
