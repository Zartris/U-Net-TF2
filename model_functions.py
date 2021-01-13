import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint

from data import *


def generate_callbacks(checkpoint_path: Path, tensorboard_path: Path, early_stop_number: int, evalGene):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=early_stop_number)
    log_dir = Path(tensorboard_path, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # log_image_dir = Path(log_dir,"images")
    # log_image_dir.mkdir(parents=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    save_model_callback = ModelCheckpoint(filepath=str(checkpoint_path),
                                          monitor='loss',
                                          verbose=2,
                                          save_best_only=True)
    # tensorboard_img_callback = ModelDiagonoser(evalGene, 1, 25, str(log_image_dir), 1, 200)
    return [save_model_callback, tensorboard_callback, early_stopping_callback]


def train_model(model: Model,
                target_size: tuple,
                batch_size: int,
                list_of_categories: list,
                train_epoch: int,
                steps_per_epoch: int,
                validation_split: float,
                train_folder: Path,
                val_folder: Path,
                checkpoint_path: Path,
                tensorboard_path: Path,
                as_gray=False,
                data_gen_args=None,
                early_stop_number=5,
                image_type='*.png'):
    if data_gen_args == None:
        data_gen_args = dict(rotation_range=90,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.05,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')

    trainGene = trainGenerator(batch_size=batch_size,
                               train_path=str(Path(train_folder, "images_sorted")),
                               image_folder='images',
                               mask_folder='label',
                               aug_dict=data_gen_args,
                               image_color_mode="rgb" if not as_gray else "grayscale",
                               mask_color_mode="rgb",
                               list_of_categories=list_of_categories,
                               save_to_dir=None,
                               target_size=target_size)
    evalGene = evalGenerator(test_path_str=str(Path(val_folder, "images_sorted")),
                             list_of_categories=list_of_categories,
                             num_image=0,
                             target_size=target_size,
                             as_gray=as_gray,
                             image_type=image_type)
    # Callbacks:
    callbacks = generate_callbacks(checkpoint_path, tensorboard_path, early_stop_number, evalGene)

    model.fit(x=trainGene,
              batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              epochs=train_epoch,
              verbose=1,
              validation_data=evalGene,
              validation_steps=40,
              callbacks=callbacks)
    return model


def test_model(model: Model,
               target_size: tuple,
               batch_size: int,
               list_of_categories: list,
               as_gray: bool,
               test_folder: Path,
               result_folder: Path,
               image_type: str,
               save_images=True,
               max_images_saved=100):
    tf.debugging.set_log_device_placement(True)

    # evalGene = evalGenerator(test_path_str=str(Path(test_folder, "images_sorted")),
    #                          list_of_categories=list_of_categories,
    #                          num_image=0,
    #                          target_size=target_size,
    #                          as_gray=as_gray,
    #                          image_type=image_type)
    #
    # eval_results = model.evaluate(x=evalGene,
    #                               batch_size=1,
    #                               verbose=1)
    # print(eval_results)

    if save_images:
        testGene = testGenerator(test_path_str=str(Path(test_folder, "images_sorted")),
                                 num_image=max_images_saved,
                                 target_size=target_size,
                                 flag_multi_class=len(list_of_categories) > 1,
                                 as_gray=as_gray,
                                 image_type=image_type)
        results = model.predict(x=testGene,
                                batch_size=batch_size,
                                verbose=1)
        # # Dry harbor save:
        DHsaveResult(save_path=str(result_folder),
                     test_path_str=str(Path(test_folder, "images_sorted")),
                     npyfile=results,
                     list_of_categories=list_of_categories,
                     image_type=image_type,
                     as_gray=as_gray,
                     add_gt=False)
        # Harbor save:
        # saveResult(save_path=str(result_folder),
        #            test_path_str=str(Path(test_folder, "images_sorted/images")),
        #            npyfile=results,
        #            list_of_categories=list_of_categories,
        #            image_type=image_type,
        #            as_gray=as_gray)

        # Segmentation predict:
        # Segmentation_help_saveResult(save_path=Path("/home/zartris/Downloads/rosbag/img_labels/temp/images_sorted/full"),
        #                              test_path_str=str(Path(test_folder, "images_sorted")),
        #                              npyfile=results,
        #                              list_of_categories=list_of_categories,
        #                              image_type=image_type,
        #                              as_gray=as_gray,
        #                              add_gt=True)


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
