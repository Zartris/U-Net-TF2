from __future__ import print_function

import glob
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage as ndi
from skimage import morphology
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Land = [221, 250, 244]
Water = [234, 165, 163]
Ship = [114, 119, 232]
Uncategorised = [0, 0, 0]

COLOR_DICT = {"land": Land, "water": Water, "ship": Ship}


def rgba2rgb(rgba, background=(255, 255, 255), to_int=True):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background
    if to_int:
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
    else:
        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B
    return np.asarray(rgb, dtype='uint8')


def show_image(img, int_to_float=False, batched=True):
    img = img.astype(int)
    if batched:
        img = img[0]
    if int_to_float:
        plt.imshow(img / 255)
    else:
        plt.imshow(img)
    plt.show()


def validate_categories(list_of_categories):
    if list_of_categories is None:
        return False
    for category in list_of_categories:
        if category[0].lower() not in COLOR_DICT:
            return False
    return True


def fill_new_mask_empty_spaces(new_mask):
    # Implement this if you need it:
    pass


def adjustData(img, mask, list_of_categories):
    new_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], len(list_of_categories)))
    new_mask = fill_new_mask(mask, new_mask, list_of_categories)
    mask = new_mask
    # cv2.imshow("test", mask[0, :, :, 0])
    # if cv2.waitKey(0):
    #     cv2.destroyAllWindows()
    return (img, mask)


def fill_new_mask(mask, new_mask, list_of_categories):
    for i, (category, color) in enumerate(list_of_categories):
        # foreach pixel in the mask, find the class in mask and convert it into one-hot vector
        index = np.where(
            np.logical_and(  # All R, G and B have to be true
                np.logical_and(
                    mask[:, :, :, 0] == color[0],
                    mask[:, :, :, 1] == color[1]
                ),
                mask[:, :, :, 2] == color[2])
        )
        index_mask = (index[0], index[1], index[2], np.zeros(len(index[0]), dtype=np.int64) + i) if (
                len(mask.shape) == 4) else (index[0], index[1], np.zeros(len(index[0]), dtype=np.int64) + i)
        new_mask[index_mask] = 1
    return new_mask


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   list_of_categories=None, save_to_dir=None, target_size=(256, 256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    if list_of_categories is None:
        list_of_categories = []
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
        seed=seed,
        interpolation='nearest')
    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        # colors = set(tuple(v) for m2d in mask[0] for v in m2d)
        if image_color_mode == "grayscale":
            img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        img, mask = adjustData(img, mask, list_of_categories)
        # for i in range(img.shape[0]):
        #     cv2.imshow("asd_" + str(i), img[i].astype(np.uint8))
        #     for j in range(len(list_of_categories)):
        #         cv2.imshow("ds_" + str(i) + "_" + str(j), mask[i, :, :, j])
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        yield (img, mask)


def evalGenerator(test_path_str,
                  list_of_categories,
                  num_image=0,
                  target_size=(256, 256),
                  as_gray=True,
                  image_type="*.png"):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    counter = 0
    if list_of_categories is None:
        list_of_categories = []
    test_path_img = Path(test_path_str, "images")
    test_path_lbl = Path(test_path_str, "label")
    test_images = []
    test_labels = []
    for png in test_path_img.glob(image_type):
        test_images.append(png)
        test_labels.append(Path(test_path_lbl, png.stem + png.suffix))
    test_images.sort()
    test_labels.sort()
    for i, png in enumerate(test_images):
        lbl = test_labels[i]
        # img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        img = io.imread(str(png))
        mask = io.imread(str(lbl))
        mask = rgba2rgb(mask)

        if not as_gray:
            img = rgba2rgb(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.resize(img, (target_size[1], target_size[0]))
        mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        if not as_gray:
            img = np.reshape(img, (1,) + img.shape)
            mask = np.reshape(mask, (1,) + mask.shape)
        else:
            img = np.reshape(img, (1,) + img.shape + (1,))
            mask = np.reshape(mask, (1,) + mask.shape)
        mask = mask.astype(np.uint8)
        img, mask = adjustData(img, mask, list_of_categories)

        # img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img

        yield (img, mask)
        if num_image != 0:
            counter += 1
            if counter == num_image:
                break
    # eval_generator = zip(image_generator, mask_generator)
    #
    # for (img, mask) in eval_generator:
    #     # colors = set(tuple(v) for m2d in mask[0] for v in m2d)
    #     if image_color_mode == "grayscale":
    #         img = img.astype(np.uint8)
    #     mask = mask.astype(np.uint8)
    #     img, mask = adjustData(img, mask, list_of_categories)
    #     # for i in range(img.shape[0]):
    #     #     cv2.imshow("asd_" + str(i), img[i].astype(np.uint8))
    #     #     for j in range(len(list_of_categories)):
    #     #         cv2.imshow("ds_" + str(i) + "_" + str(j), mask[i, :, :, j])
    #     # if cv2.waitKey(0):
    #     #     cv2.destroyAllWindows()
    #     yield (img, mask)


def testGenerator(test_path_str, num_image=0, target_size=(256, 256), flag_multi_class=False,
                  as_gray=True,
                  image_type="*.png"):
    test_path_img = Path(test_path_str, "images")
    counter = 0
    test_images = []
    for png in test_path_img.glob(image_type):
        test_images.append(png)
    test_images.sort()
    for i, png in enumerate(test_images):
        # img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        img = io.imread(str(png))
        if not as_gray:
            img = rgba2rgb(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.resize(img, (target_size[1], target_size[0]))
        # img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        if not as_gray:
            img = np.reshape(img, (1,) + img.shape)
        else:
            img = np.reshape(img, (1,) + img.shape + (1,))
        yield img

        if num_image != 0:
            counter += 1
            if counter == num_image:
                break


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image",
                 mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
                         as_gray=mask_as_gray)
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


def predictionToMask(list_of_categories, img, inv=False):
    # img = np.argmax(img, axis=2) if len(img.shape) == 3 else img
    if len(list_of_categories) > 1:
        img = np.argmax(img, axis=2)
        img_out = np.zeros((img.shape[0], img.shape[1], 3))
        for i, (name, color) in enumerate(list_of_categories):
            img_out[img == i] = color
    else:
        img_out = np.zeros(img.shape)
        if inv:
            img_out[img <= 0.5] = 1
        else:
            img_out[img > 0.5] = 1

    return img_out


def combine_images(images: list):
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]
    result_image = images[0]
    for i in range(1, len(images)):
        result_image = cv2.hconcat([result_image, images[i]])

    return result_image


def do_text(image, p, offset, text):
    pos = (int(p[0]) + int(offset[0]), int(p[1]) + int(offset[1]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 0, 255)
    line_type = 2
    cv2.putText(image, str(text), pos, font, font_scale, font_color, line_type)


def Segmentation_help_saveResult(save_path: Path, test_path_str, npyfile, list_of_categories, image_type, as_gray,
                                 add_gt=False):
    if save_path.exists():
        shutil.rmtree(str(save_path))
    save_path.mkdir(parents=True)
    original_images = []
    img_path = Path(test_path_str, "images")
    for png in img_path.glob(image_type):
        original_images.append(png)
    original_images.sort()

    for i, item in enumerate(npyfile):
        org_img_path = original_images[i]
        label_name = org_img_path.stem + "_color_mask" + org_img_path.suffix
        org_image = cv2.imread(str(org_img_path), cv2.IMREAD_COLOR)
        oh, ow = org_image.shape[:2]

        resulting_mask = item_to_mask(item, list_of_categories)
        resulting_mask = cv2.resize(resulting_mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        resulting_mask = cv2.cvtColor(resulting_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(save_path, label_name)), resulting_mask)
        cv2.imwrite(str(Path(save_path, org_img_path.stem + org_img_path.suffix)), org_image)
        print("saving:", str(Path(save_path, org_img_path.stem + org_img_path.suffix)))


def DHsaveResult(save_path, test_path_str, npyfile, list_of_categories, image_type, as_gray, add_gt=False):
    original_images = []
    original_labels = []
    img_path = Path(test_path_str, "images")
    label_path = Path(test_path_str, "label")
    for png in img_path.glob(image_type):
        original_images.append(png)
    if add_gt:
        for png in label_path.glob(image_type):
            original_labels.append(png)
    original_images.sort()
    original_labels.sort()

    images_processed = []
    current_index = 0
    for i, item in enumerate(npyfile):
        h, w = item.shape[:2]
        org_img_path = original_images[i]
        if as_gray:
            org_image = cv2.imread(str(org_img_path), cv2.IMREAD_GRAYSCALE)
            org_image = cv2.merge([org_image, org_image, org_image])
        else:
            org_image = cv2.imread(str(org_img_path), cv2.IMREAD_COLOR)
        org_image = cv2.resize(org_image, dsize=(item.shape[1], item.shape[0]))
        # item = cv2.blur(item, (5, 5))

        resulting_mask = item_to_mask(item, list_of_categories)
        do_text(resulting_mask, (h, int(w / 2)), (-30, 0), "Prediction")
        org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
        combined = [org_image]
        if add_gt:
            org_label = cv2.imread(str(original_labels[i]), cv2.IMREAD_COLOR)
            org_label = cv2.resize(org_label, dsize=(item.shape[1], item.shape[0]))
            org_label = cv2.cvtColor(org_label, cv2.COLOR_RGB2BGR)
            do_text(org_label, (h, int(w / 2)), (-30, 0), "Ground truth")
            combined.append(org_label)
        combined.append(resulting_mask)
        for j, (name, color) in enumerate(list_of_categories):
            c_img = item[:, :, j] * 255
            c_img = c_img.astype(np.uint8)
            c_img = cv2.merge([c_img, c_img, c_img])
            do_text(c_img, (h, int(w / 2)), (-30, 0), name)
            combined.append(c_img)
        combined_img = combine_images(combined)
        print("saving:", os.path.join(save_path, "%d_predict_all.png" % i))
        io.imsave(os.path.join(save_path, "%d_predict_all.png" % i), combined_img)


def saveResult(save_path, test_path_str, npyfile, list_of_categories, image_type, as_gray):
    original_images = []
    for png in Path(test_path_str).glob(image_type):
        original_images.append(png)
    original_images.sort()
    images_processed = []
    current_index = 0
    for i, item in enumerate(npyfile):
        org_path = original_images[i]
        if as_gray:
            org_image = cv2.imread(str(org_path), cv2.IMREAD_GRAYSCALE)
            org_image = cv2.merge([org_image, org_image, org_image])
        else:
            org_image = cv2.imread(str(org_path), cv2.IMREAD_COLOR)
        org_image = cv2.resize(org_image, dsize=(item.shape[1], item.shape[0]))
        # item = cv2.blur(item, (5, 5))

        threshed_item = adaptive_threshold(item)
        o_th_item = otsu_filter(item)
        img = item_to_mask(item, list_of_categories)
        fill_holes = item_to_mask_fill_holse(item, list_of_categories)
        img_stack = item
        for img_p in images_processed:
            img_b = cv2.blur(img_p, (13, 13))
            img_stack = img_stack + np.reshape(img_b, img_stack.shape)
        img_stack = img_stack / (len(images_processed) + 1)
        img_stack = item_to_mask(img_stack, list_of_categories)
        if len(images_processed) == 2:
            images_processed.pop(0)
        images_processed.append(item)
        # images_processed.insert(current_index, item)
        # current_index = (current_index + 1) % 1

        img_item = item * 255
        org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
        img_item_uint8 = img_item.astype(np.uint8)
        img_item_uint8 = cv2.merge([img_item_uint8, img_item_uint8, img_item_uint8])

        combined_img = combine_images([org_image, img, fill_holes, img_item_uint8, threshed_item, o_th_item, img_stack])
        print("saving:", os.path.join(save_path, "%d_predict_all.png" % i))
        io.imsave(os.path.join(save_path, "%d_predict_all.png" % i), combined_img)


def item_to_mask_fill_holse(item, list_of_categories):
    img = predictionToMask(list_of_categories, item, inv=True)
    img = np.reshape(img, (img.shape[0], img.shape[1])).astype(np.bool)
    img = morphology.remove_small_objects(img, min_size=100)
    img = morphology.binary_dilation(img)
    img = ndi.binary_fill_holes(img, structure=morphology.disk(5))
    img = img * 255
    img = cv2.merge([img, img, img])
    return img.astype(np.uint8)


def item_to_mask(item, list_of_categories):
    img = predictionToMask(list_of_categories, item)
    if len(list_of_categories) == 1:
        img = img * 255
        img = cv2.merge([img, img, img])
    return img.astype(np.uint8)


def otsu_filter(item):
    threshed_item = item * 255
    # threshed_item = cv2.merge([threshed_item, threshed_item, threshed_item])
    threshed_item = threshed_item.astype(np.uint8)
    blur = cv2.GaussianBlur(threshed_item, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th3 = cv2.merge([th3, th3, th3])
    return th3


def adaptive_threshold(item):
    threshed_item = item * 255
    # threshed_item = cv2.merge([threshed_item, threshed_item, threshed_item])
    threshed_item = threshed_item.astype(np.uint8)
    threshed_item = cv2.medianBlur(threshed_item, 9)
    # threshed_item = cv2.adaptiveThreshold(threshed_item, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
    #                                       2)
    threshed_item = cv2.adaptiveThreshold(threshed_item, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7,
                                          2)
    threshed_item = cv2.merge([threshed_item, threshed_item, threshed_item])
    return threshed_item


if __name__ == '__main__':
    # Test validate classes
    loc = ["land", "water"]
    assert validate_categories(loc)
    loc.append("pig")
    assert not validate_categories(loc)
