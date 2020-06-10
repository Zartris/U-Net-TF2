import os

import io
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer


def make_image_tensor(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Adapted from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 4:
        return tensor
    if len(tensor.shape) == 3:
        height, width, channel = tensor.shape
        return np.reshape(tensor, (-1, height, width, channel))
    if len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
        return np.reshape(tensor, (-1, height, width, channel))
    sys.exit("Wrong tensor size")


class TensorboardWriter:

    def __init__(self, outdir):
        assert (os.path.isdir(outdir))
        self.outdir = str(outdir)
        self.writer = tf.summary.create_file_writer(self.outdir, flush_millis=1000)

    def save_image(self, tag, image, global_step=None):
        image_tensor = make_image_tensor(image)
        tf.summary.image(name=tag, data=image_tensor, step=global_step)
        self.writer.flush()
        # self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, image=image_tensor)]),
        #                         global_step)

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()


class ModelDiagonoser(Callback):

    def __init__(self, data_generator, batch_size, num_samples, output_dir, normalization_mean):
        super().__init__()
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.tensorboard_writer = TensorboardWriter(output_dir)
        self.normalization_mean = normalization_mean
        is_sequence = isinstance(self.data_generator, Sequence)
        if is_sequence:
            self.enqueuer = OrderedEnqueuer(self.data_generator,
                                            use_multiprocessing=True,
                                            shuffle=False)
        else:
            self.enqueuer = GeneratorEnqueuer(self.data_generator,
                                              use_multiprocessing=True)
        self.enqueuer.start(workers=1, max_queue_size=4)

    def on_epoch_end(self, epoch, logs=None):
        output_generator = self.enqueuer.get()
        steps_done = 0
        total_steps = int(np.ceil(np.divide(self.num_samples, self.batch_size)))
        sample_index = 0
        while steps_done < total_steps:
            generator_output = next(output_generator)
            x, y = generator_output[:2]
            y_pred = self.model.predict(x)
            y_pred = np.argmax(y_pred, axis=-1)
            y_true = np.argmax(y, axis=-1)

            for i in range(0, len(y_pred)):
                n = steps_done * self.batch_size + i
                if n >= self.num_samples:
                    return
                img = np.squeeze(x[i, :, :, :])
                img = 255. * (img + self.normalization_mean)  # mean is the training images normalization mean
                img = img[:, :, [2, 1, 0]]  # reordering of channels

                pred = y_pred[i]
                pred = pred.reshape(img.shape[0:2])

                ground_truth = y_true[i]
                ground_truth = ground_truth.reshape(img.shape[0:2])

                self.tensorboard_writer.save_image("Epoch-{}/{}/x"
                                                   .format(self.epoch_index, sample_index), img)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y"
                                                   .format(self.epoch_index, sample_index), ground_truth)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y_pred"
                                                   .format(self.epoch_index, sample_index), pred)
                sample_index += 1

            steps_done += 1

    def on_train_end(self, logs=None):
        self.enqueuer.stop()
        self.tensorboard_writer.close()
