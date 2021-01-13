import argparse
import sys

import tensorflow as tf

from data import *
from model_functions import train_model, test_model
from models.UNetModel import UNetBinary, UNetSmall, UNet


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


def bgr_to_rgb(categories):
    result = []
    for (name, (B, G, R)) in categories:
        result.append((name, (R, G, B)))
    return result


if __name__ == '__main__':
    # Note to aspect ratio : You want to keep aspect ratio of 4:3. And your network wants to divide it to 2 for a while.
    # So 4*2*2*2*2*2*2*2 = 512, 3*2*2*2*2*2*2*2=384
    # Mine is 16/9 ratio: 16*2*2*2*2*2*2=1024  9*2*2*2*2*2*2 = 576

    # Setting up gpu's
    setup_gpus()

    # Input parameters:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",
                        default=5,
                        type=int)  # The seed for testing
    parser.add_argument("--epochs",
                        default=100,
                        type=int)  # Number of episodes to train for
    parser.add_argument("--steps_per_epoch",
                        default=500,
                        type=int)  # number of batches trained on per epoch call
    parser.add_argument("--batch_size",
                        default=3,
                        type=int)  # Batch size for training
    parser.add_argument("--image_width",
                        default=16 * 2 * 2 * 2 * 2,
                        type=int,
                        help="Your network wants to divide it to 2 for a while.\\"
                             "So a tip is to find the aspect ration you need and mulitply until you have a sufficient width\\"
                             "example: So 4*2*2*2*2*2*2*2 = 512, 3*2*2*2*2*2*2*2=384")  # image width
    parser.add_argument("--image_height",
                        default=9 * 2 * 2 * 2 * 2,
                        type=int,
                        help="Your network wants to divide it to 2 for a while.\\"
                             "So a tip is to find the aspect ration you need and mulitply until you have a sufficient height\\"
                             "example: So 4*2*2*2*2*2*2*2 = 512, 3*2*2*2*2*2*2*2=384")  # image height
    parser.add_argument("--image_channels",
                        default=3,
                        type=int,
                        help="1= grayscale, 3=rgb, 4=rgba")  # image channels
    parser.add_argument("--lr",
                        default=1e-4,
                        type=float)  # Learning rate
    parser.add_argument("--early_stop",
                        default=5,
                        type=int)  # Learning rate
    parser.add_argument("--weight_decay",
                        default=1e-6)  # weight_decay
    parser.add_argument("--load_model_path",
                        default="")  # If should load model: if "" don't load anything
    parser.add_argument("--eval",
                        action='store_true')  # If we only want to evaluate a model.
    parser.add_argument("--cpu",
                        action='store_true')  # If we only want to evaluate a model.
    parser.add_argument("--jpg",
                        action='store_true')  # If we only want to evaluate a model.
    parser.add_argument("--model_prefix",
                        default="",
                        type=str)  # Learning rate

    args = parser.parse_args()
    image_type = "*.jpg" if args.jpg else "*.png"

    # Hard-coded values
    load_model = True
    # BGR
    categories = [("sky", (207, 91, 108)),
                  ("object", (239, 213, 155)),
                  ("dock_side", (224, 141, 173)),
                  ("floor", (244, 250, 221)),
                  ("ship", (232, 119, 114)),
                  ("unknown", (130, 219, 128))
                  ]

    # categories = [("water", (180, 130, 70)),
    #               ("ship", (35, 142, 107)),
    #               ("unknown", (81, 0, 81))]

    # RGB
    categories = bgr_to_rgb(categories)

    nclasses = len(categories)
    validation_split = 0.2

    input_shape = (args.batch_size, args.image_height, args.image_width, args.image_channels)
    target_size = (args.image_height, args.image_width)

    as_gray = args.image_channels == 1
    small = True
    if as_gray:
        model_name = 'unet' + "_" + str(args.image_width) + "_" + str(
            args.image_height) + str(args.model_prefix) + '_grayscale_multi_small.hdf5'
    elif small:
        model_name = 'unet' + "_" + str(args.image_width) + "_" + str(args.image_height) + str(
            args.model_prefix) + '_rgb_multi_small.hdf5'
    else:
        model_name = 'unet' + "_" + str(args.image_width) + "_" + str(args.image_height) + str(
            args.model_prefix) + '_rgb_multi_test123.hdf5'

    # DATA PATH
    current_dir = os.getcwd()

    # data_folder = Path(current_dir, "..", "Segmentation_Data")
    data_folder = Path("C:\\Users\\Jonas le Fevre\\Documents\\AirSim\\Dry_small\\combined")
    # data_folder = Path("/media/zartris/VOID/code/python/Airsim/Segmentation_Data/dry_harbor")
    train_folder = Path(data_folder, "train")
    # test_folder = Path(data_folder, "test")
    val_folder = Path(data_folder, "test")
    test_folder = Path(data_folder, "gazebo_sample_images")
    tensorboard_path = Path(data_folder, "tensorboard")

    result_folder = Path(data_folder, "result", model_name.split(".")[0])
    if result_folder.exists():
        rm_tree(result_folder)
    result_folder.mkdir(parents=True)

    checkpoint_dir = Path(current_dir, "saved_model")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    # Choose an optimizer and loss function for training:
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9, decay=0.001)
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    # loss_object = tf.keras.losses.MeanSquaredError()
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr)

    # Create model:
    if nclasses == 1:
        model = UNetBinary()
    elif small:
        model = UNetSmall(nclasses=nclasses)
    else:
        model = UNet(nclasses=nclasses)

    model.compile(optimizer=optimizer, loss=loss_object,
                  metrics=[tf.keras.metrics.CategoricalCrossentropy(), 'accuracy'])
    model.build(input_shape=input_shape)

    checkpoint_path = Path(checkpoint_dir, model_name)
    if args.load_model_path != "":
        # Loading a specific model
        checkpoint_path = Path(args.load_model_path)
        load_model = True
        if not checkpoint_path.exists():
            sys.exit("The path specified is not existing: " + str(checkpoint_path))

    # Check if we should load a model:
    if load_model and checkpoint_path.exists():
        print("Loading old weights from:", str(checkpoint_path))
        model.load_weights(str(checkpoint_path))

    data_gen_args = dict(rotation_range=0,
                         width_shift_range=50,
                         height_shift_range=50,
                         shear_range=0,  # If semantic segmentation set these to 0
                         zoom_range=0,  # If semantic segmentation set these to 0
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest')
    print(model.summary())
    if args.cpu:
        with tf.device('/cpu:0'):
            print(tf.device)
            if not args.eval:  # Training
                model = train_model(model=model,
                                    target_size=target_size,
                                    batch_size=args.batch_size,
                                    list_of_categories=categories,
                                    train_epoch=args.epochs,
                                    steps_per_epoch=args.steps_per_epoch,
                                    validation_split=validation_split,
                                    train_folder=train_folder,
                                    val_folder=val_folder,
                                    checkpoint_path=checkpoint_path,
                                    tensorboard_path=tensorboard_path,
                                    as_gray=as_gray,
                                    data_gen_args=data_gen_args,
                                    early_stop_number=args.early_stop,
                                    image_type=image_type)
                print("Done training")

            model.load_weights(str(checkpoint_path))
            # else just eval
            test_model(model=model,
                       target_size=target_size,
                       batch_size=args.batch_size,
                       list_of_categories=categories,
                       as_gray=as_gray,
                       test_folder=test_folder,
                       result_folder=result_folder,
                       image_type=image_type,
                       save_images=True,
                       max_images_saved=0)
    else:
        print(tf.device)
        if not args.eval:  # Training
            model = train_model(model=model,
                                target_size=target_size,
                                batch_size=args.batch_size,
                                list_of_categories=categories,
                                train_epoch=args.epochs,
                                steps_per_epoch=args.steps_per_epoch,
                                validation_split=validation_split,
                                train_folder=train_folder,
                                val_folder=val_folder,
                                checkpoint_path=checkpoint_path,
                                tensorboard_path=tensorboard_path,
                                as_gray=as_gray,
                                data_gen_args=data_gen_args,
                                early_stop_number=args.early_stop,
                                image_type=image_type)
            print("Done training")
        model.load_weights(str(checkpoint_path))
        # else just eval
        test_model(model=model,
                   target_size=target_size,
                   batch_size=args.batch_size,
                   list_of_categories=categories,
                   as_gray=as_gray,
                   test_folder=test_folder,
                   result_folder=result_folder,
                   image_type=image_type,
                   save_images=True,
                   max_images_saved=0)
