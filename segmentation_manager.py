import sys
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
import tensorflow as tf

from data import rgba2rgb
from models.UNetModel import UNetBinary, UNet


class SegmentationManager(ABC):
    def __init__(self, is_grayscale: bool, checkpoint_path: str, cpu=False):
        self.model = None
        self.cpu = cpu
        self.setup_gpus()
        self.is_grayscale = is_grayscale
        path = Path(checkpoint_path)
        if not path.exists():
            sys.exit("The checkpoint path specified is not existing: " + str(checkpoint_path))
        self.setup(path)

    @abstractmethod
    def run(self, image):
        pass

    @abstractmethod
    def setup(self, checkpoint_path: Path):
        """
        Need to set all the self variables.
        :return:
        """
        pass

    @staticmethod
    def setup_gpus():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    @abstractmethod
    def prepare_image(self, img):
        pass


class UNetManager(SegmentationManager):
    def __init__(self, input_shape, is_grayscale: bool, checkpoint_path: str, cpu=False, number_of_classes=1):
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes
        super().__init__(is_grayscale, checkpoint_path, cpu)

    def run(self, image):
        """
        Give full size image and we will prepare the data for you
        :return:
        """
        img = self.prepare_image(image)
        # cv2.imshow("before", img[0])
        if self.cpu:
            with tf.device('/cpu:0'):
                result = self.process(img)
        else:
            result = self.process(img)
        result = self.prepare_result(result)
        return result

    def process(self, img):
        return self.model.predict(img)

    def prepare_result(self, result):
        return result

    def setup(self, checkpoint_path: Path):
        if self.number_of_classes > 1:
            self.model = UNet(nclasses=self.number_of_classes)
        else:
            self.model = UNetBinary()

        # Not used, but needed to compile
        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.RMSprop()

        self.model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
        self.model.build(input_shape=self.input_shape)

        print("Loading weights from:", str(checkpoint_path))
        self.model.load_weights(str(checkpoint_path))

        self.warmup()

    def warmup(self):
        path = "G:\\code\\python\\Airsim\\Segmentation_Data\\WarmupImage\\img_down_center_custom_1592210349404086000.png"
        img = io.imread(str(path))
        self.run(img)

    def prepare_image(self, img):
        if not self.is_grayscale:
            img = rgba2rgb(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
        # cv2.imshow("img", img)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        # img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        if not self.is_grayscale:
            img = np.reshape(img, (1,) + img.shape)
        else:
            img = np.reshape(img, (1,) + img.shape + (1,))
        return img
