import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint

from data import *


def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def setup_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def generate_callbacks(checkpoint_path: Path, tensorboard_path: Path):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    log_dir = Path(tensorboard_path, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    save_model_callback = ModelCheckpoint(filepath=str(checkpoint_path),
                                          monitor='loss',
                                          verbose=1,
                                          save_best_only=True)
    return [save_model_callback, tensorboard_callback, early_stopping_callback]


def train_model(model: Model,
                target_size: tuple,
                batch_size: int,
                list_of_categories: list,
                train_epoch: int,
                steps_per_epoch: int,
                validation_split: float,
                train_folder: Path,
                checkpoint_path: Path,
                tensorboard_path: Path):
    # Callbacks:
    callbacks = generate_callbacks(checkpoint_path, tensorboard_path)
    data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest',
                         validation_split=validation_split)

    trainGene = trainGenerator(batch_size=batch_size,
                               train_path=str(Path(train_folder, "images_sorted")),
                               image_folder='images',
                               mask_folder='label',
                               aug_dict=data_gen_args,
                               image_color_mode="rgb",
                               mask_color_mode='rgb',
                               list_of_categories=list_of_categories,
                               save_to_dir=None,
                               target_size=target_size)
    model.fit(x=trainGene,
              batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              epochs=train_epoch,
              callbacks=callbacks)
    return model


def test_model(model: Model,
               target_size: tuple,
               batch_size: int,
               list_of_categories: list,
               as_gray: bool,
               test_folder: Path,
               result_folder: Path):
    testGene = testGenerator(test_path_str=str(Path(test_folder, "images_sorted/images")),
                             num_image=0,
                             target_size=target_size,
                             flag_multi_class=len(list_of_categories) > 1,
                             as_gray=as_gray)

    results = model.predict(x=testGene,
                            batch_size=batch_size,
                            verbose=1)

    saveResult(save_path=str(result_folder),
               npyfile=results,
               list_of_categories=list_of_categories)


if __name__ == '__main__':
    pass
    # # Setting up gpu's
    # setup_gpus()
    # # Not to aspect ratio : You want to keep aspect ratio of 4:3. And your network wants to divide it to 2 for a while.
    # # So 4*2*2*2*2*2*2*2 = 512, 3*2*2*2*2*2*2*2=384
    # WIDTH = int(512)
    # HEIGHT = int(384)
    # CHANNELS = 3
    # BATCH_SIZE = 4
    # input_shape = (BATCH_SIZE, WIDTH, HEIGHT, CHANNELS)
    # nclasses = 3
    # load_previous = True
    # as_gray = False
    # model_name = 'unet' + "_" + str(WIDTH) + "_" + str(HEIGHT) + '.hdf5'
    # # DATA PATH
    # dir = os.getcwd()
    #
    # data_folder = Path(dir, "Data")
    # train_folder = Path(data_folder, "train")
    # test_folder = Path(data_folder, "test")
    # result_folder = Path(test_folder, "result")
    # if result_folder.exists():
    #     rm_tree(result_folder)
    # result_folder.mkdir(parents=True)
    # checkpoint_dir = Path(dir, "saved_model")
    # if not checkpoint_dir.exists():
    #     checkpoint_dir.mkdir(parents=True)
    # checkpoint_path = Path(checkpoint_dir, model_name)
    #
    # # Choose an optimizer and loss function for training:
    # loss_object = tf.keras.losses.CategoricalCrossentropy()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    #
    # # Create model:
    # model = UNet(nclasses=3)
    # model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
    # model.build(input_shape=input_shape)
    #
    # # Binaries the labels to only detect water:
    # if load_previous and checkpoint_path.exists():
    #     print("Loading old weights from:", str(checkpoint_path))
    # model.load_weights(str(checkpoint_path))
    #
    # train_model(model, (WIDTH, HEIGHT), BATCH_SIZE, nclasses, 1000, 1000, 0.2, train_folder, checkpoint_path)
    # test_model(model, (WIDTH, HEIGHT), BATCH_SIZE, nclasses, test_folder)
