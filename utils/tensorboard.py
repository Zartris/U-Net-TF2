import io
import itertools
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer
import sklearn

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

    def save_image(self, tag, image, global_step):
        image_tensor = make_image_tensor(image)
        with self.writer.as_default():
            tf.summary.image(name=tag, data=image_tensor, step=global_step)

        # self.writer.flush()
        # self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, image=image_tensor)]),
        #                         global_step)

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()


class LogConfusionMatrix(Callback):
    def __init__(self, log_dir, data_generator, class_names):
        super().__init__()
        self.data_generator = data_generator
        self.file_writer_cm = tf.summary.create_file_writer(Path(log_dir, 'cm'))
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        test_images, test_labels = next(self.data_generator)
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(test_images)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure = self.plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = self.plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    def plot_to_image(self,figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

class ModelDiagonoser(Callback):

    def __init__(self, data_generator, batch_size, num_samples, output_dir, normalization_mean, start_index=0):
        super().__init__()
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.tensorboard_writer = TensorboardWriter(output_dir)
        self.normalization_mean = normalization_mean
        self.start_index = start_index
        is_sequence = isinstance(self.data_generator, Sequence)
        if is_sequence:
            self.enqueuer = OrderedEnqueuer(self.data_generator,
                                            use_multiprocessing=False,
                                            shuffle=False)
        else:
            self.enqueuer = GeneratorEnqueuer(self.data_generator,
                                              use_multiprocessing=False)
        self.enqueuer.start(workers=1, max_queue_size=4)

    def on_epoch_end(self, epoch, logs=None):
        steps_done = 0
        total_steps = int(np.ceil(np.divide(self.num_samples, self.batch_size)))
        sample_index = 0
        while steps_done < total_steps:
            x, y = next(self.data_generator)
            sample_index += 1
            if sample_index <= self.start_index:
                continue
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
                                                   .format(epoch, sample_index - 1), img, epoch)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y"
                                                   .format(epoch, sample_index - 1), ground_truth, epoch)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y_pred"
                                                   .format(epoch, sample_index - 1), pred, epoch)
                sample_index += 1

            steps_done += 1

    def on_train_end(self, logs=None):
        self.enqueuer.stop()
        self.tensorboard_writer.close()
