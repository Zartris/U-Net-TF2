from __future__ import print_function

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.transform as trans
from PIL.Image import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Land = [221, 250, 244]
Water = [221, 199, 220]
Ship = [114, 119, 232]

COLOR_DICT = np.array([Land, Water, Ship])


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def show_image(w, h):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:256, 0:256] = [255, 0, 0]  # red patch in upper left
    img = Image.fromarray(data, 'RGB')
    img.save('my.png')
    img.show()


def adjustData(img, mask, flag_multi_class, num_class):
    if (flag_multi_class):
        ## TODO :: Work here needs to be done!
        # plt.imshow(mask[0] / 255)
        # plt.show()
        img = img / 255
        # mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], num_class))
        for i in range(num_class):
            # foreach pixel in the mask, find the class in mask and convert it into one-hot vector
            index = np.where(
                np.logical_and(  # All R G and B have to be true
                    np.logical_and(
                        mask[:, :, :, 0] == COLOR_DICT[i][0],
                        mask[:, :, :, 1] == COLOR_DICT[i][1]
                    ),
                    mask[:, :, :, 2] == COLOR_DICT[i][2]))

            index_mask = (index[0], index[1], index[2], np.zeros(len(index[0]), dtype=np.int64) + i) if (
                    len(mask.shape) == 4) else (index[0], index[1], np.zeros(len(index[0]), dtype=np.int64) + i)
            new_mask[index_mask] = 1

        mask = new_mask

    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def testGenerator(test_path_str, num_image=0, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    test_path = Path(test_path_str)
    counter = 0
    for png in test_path.glob("*.png"):
        img = io.imread(str(png), as_gray=as_gray)
        img = rgba2rgb(img)
        img = img / 255.
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img

        if num_image != 0:
            counter += 1
            if counter == num_image:
                break


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out


def predictionToMask(color_dict, img):
    img = np.argmax(img, axis=2) if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i, rgb in enumerate(color_dict):
        img_out[img == i] = rgb
    return img_out


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = predictionToMask(COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        img_uint8 = img.astype(np.uint8)
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img_uint8)
