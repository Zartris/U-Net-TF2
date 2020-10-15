"""
 * Python script to demonstrate Canny edge detection.
 *
 * usage: python CannyEdge.py <filename> <sigma> <low_threshold> <high_threshold>
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, io, segmentation, measure, draw
from skimage import morphology
from skimage.filters import sobel, gaussian
from skimage.transform import probabilistic_hough_line


def gaussian_blur(image, sigma=1.0):
    return gaussian(image, sigma, preserve_range=True)


def image_contours(image, level=0.8, show=True):
    contours = measure.find_contours(image, level=level, fully_connected='high', positive_orientation='low')
    # Display the image and plot all contours found
    if show:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.gray())
        for n, contour in enumerate(contours):
            draw.set_color(image, draw.polygon(contour[:, 0], contour[:, 1]), True)
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax.set_title("Contours")
        ax.axis_order('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    return image


def show_plot(image, name):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_title(name)
    ax.axis_order('off')
    plt.show()


def canny_img(image, sigma=1.0, low_th=None, high_th=None, show=True):
    edges = feature.canny(image, sigma=sigma, low_threshold=low_th, high_threshold=high_th)
    if show:
        show_plot(edges, "Canny")
    # edges = np.array(edges)
    # edges = edges.astype(np.float)
    return edges


def edge_based_segmentation(image):
    start = time.perf_counter()
    image = gaussian_blur(image, sigma=0.5)
    # image = np.array(image).astype(np.float)
    # image = image / np.max(image)
    # show_plot(image, "after blur")
    canny = canny_img(image, sigma=0.6, low_th=0.09, high_th=0.1, show=True)

    canny = morphology.binary_dilation(canny, selem=morphology.disk(2))
    show_plot(canny, "Dilation")
    canny = morphology.remove_small_objects(canny, min_size=100)
    show_plot(canny, "remove small objects")

    canny = morphology.binary_closing(canny, selem=morphology.disk(3))
    canny = morphology.binary_closing(canny, selem=morphology.disk(10))
    show_plot(canny, "Closing")
    canny = ndi.binary_fill_holes(canny)
    show_plot(canny, "Binary holes filled")

    contour_image = image_contours(canny, level=0.50, show=True)
    show_plot(contour_image, "test")
    contour_image = morphology.remove_small_objects(contour_image, min_size=100)

    contour_image = ndi.binary_fill_holes(contour_image)
    show_plot(contour_image, "before test")
    contour_image = morphology.binary_closing(contour_image, selem=morphology.disk(5))
    contour_image = np.pad(contour_image, pad_width=1, mode='constant', constant_values=1)

    show_plot(contour_image, "test")
    # contour_image = morphology.binary_dilation(contour_image)

    contour_image = image_contours(contour_image, level=0.50, show=True)
    show_plot(contour_image, "test after")
    # Combine images:
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray, interpolation='none')
    ax.imshow(contour_image, cmap=plt.cm.jet, interpolation='none', alpha=0.3)
    ax.set_title('finished')
    ax.axis_order('off')
    plt.show()
    #
    # plt.figure()
    # plt.imshow(image, 'gray', interpolation='none')
    # plt.imshow(contour_image, 'jet', interpolation='none', alpha=0.3)
    # plt.title("Finished")
    # plt.axis('off')
    # plt.show()
    stop = time.perf_counter()
    print(f"process images: {stop - start:0.4f} seconds")


def hough_line(org_image, work_image, threshold=10, line_length=50, line_gap=10, show=True):
    lines = probabilistic_hough_line(work_image, threshold=threshold, line_length=line_length,
                                     line_gap=line_gap)
    fig, ax = plt.subplots()
    ax.imshow(org_image, cmap=plt.gray())
    for line in lines:
        p0, p1 = line
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax.set_title("hough lines")
    ax.axis_order('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def region_based_segmentation(image):
    elevation_map = sobel(image)
    show_plot(elevation_map, "elevation_map")
    markers = find_makers(image)
    segmentation_coins = segmentation.watershed(elevation_map, markers)
    show_plot(segmentation_coins, name="Segmentation")


def find_makers(image, low_th=30, high_th=150, show=True):
    markers = np.zeros_like(image)
    markers[image * 255 < low_th] = 1
    markers[image * 255 > high_th] = 2
    if show:
        fig, ax = plt.subplots()
        ax.imshow(markers, cmap=plt.cm.nipy_spectral)
        ax.set_title('markers')
        ax.axis_order('off')
    return markers


if __name__ == '__main__':
    steps = 100
    # 5 is corrupt
    target = 5


    # read command-line arguments
    input_folder = Path("/home/zartris/Downloads/rosbag/extracted_thermal")
    sigma = 0.60
    low_threshold = 0.09
    high_threshold = 0.2
    for index, filename in enumerate(input_folder.glob("*.jpg")):
        if 0 <= target == index:
            break
        if target >= 0 and target != index + 1:
            continue
        if index >= steps:
            break

        # load and display original image as grayscale
        image = io.imread(fname=str(filename), as_gray=True)
        show_plot(image, "original")
        edge_based_segmentation(image)
        # region_based_segmentation(image)

test_s = 0.60
test_lt = 0.09
test_ht = 0.2
