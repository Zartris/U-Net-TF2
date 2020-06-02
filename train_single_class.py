import datetime

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

from data import *
from models.UNetModel import UNet


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


if __name__ == '__main__':
    # Setting up gpu's
    setup_gpus()
    WIDTH = int(1280 / 3)
    HEIGHT = int(720 / 3)
    CHANNELS = 3
    BATCH_SIZE = 1
    input_shape = (BATCH_SIZE, WIDTH, HEIGHT, CHANNELS)
    nclasses = 3
    load_previous = True
    as_gray = False
    model_name = 'unet' + "_" + str(WIDTH) + "_" + str(HEIGHT) + '.hdf5'
    # DATA PATH
    dir = os.getcwd()

    data_folder = Path(dir, "Data")
    train_folder = Path(data_folder, "train")
    test_folder = Path(data_folder, "test")
    result_folder = Path(test_folder, "result")
    if result_folder.exists():
        rm_tree(result_folder)
    result_folder.mkdir(parents=True)
    checkpoint_dir = Path(dir, "saved_model")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    # Choose an optimizer and loss function for training:
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    # Create model:
    model = UNet(nclasses=3)
    model.compile(optimizer, loss_object, metrics=['accuracy'])
    model.build(input_shape)
    # Binaries the labels to only detect water:
    if load_previous and Path(checkpoint_dir, model_name).exists():
        print("Loading old weights from:", str(Path(checkpoint_dir, model_name)))
        model.load_weights(str(Path(checkpoint_dir, model_name)))

    log_dir = Path(dir, "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    data_gen_args = dict(rotation_range=0.9,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    trainGene = trainGenerator(BATCH_SIZE,
                               train_path='Data/train/images_sorted',
                               image_folder='images',
                               mask_folder='label',
                               aug_dict=data_gen_args,
                               image_color_mode="rgb",
                               mask_color_mode='rgb',
                               flag_multi_class=True,
                               num_class=nclasses,
                               save_to_dir=None,
                               target_size=(WIDTH, HEIGHT))

    model_checkpoint = ModelCheckpoint(str(Path(checkpoint_dir, model_name)), monitor='loss', verbose=1,
                                       save_best_only=True)
    model.fit(trainGene, batch_size=None, steps_per_epoch=100, epochs=5,
              callbacks=[model_checkpoint, tensorboard_callback])

    testGene = testGenerator(test_path_str="Data/test/images_sorted/images",
                             num_image=50,
                             target_size=(WIDTH, HEIGHT),
                             flag_multi_class=True,
                             as_gray=as_gray)
    results = model.predict(testGene, batch_size=None, verbose=1)

    saveResult(str(result_folder), results, True, 3)
