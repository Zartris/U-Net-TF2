import argparse

import tensorflow as tf

from data import *
from model_functions import train_model, test_model
from models.UNetModel import UNetBinary


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
    # Note to aspect ratio : You want to keep aspect ratio of 4:3. And your network wants to divide it to 2 for a while.
    # So 4*2*2*2*2*2*2*2 = 512, 3*2*2*2*2*2*2*2=384

    # Setting up gpu's
    setup_gpus()

    # Input parameters:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",
                        default=0,
                        type=int)  # The seed for testing
    parser.add_argument("--epochs",
                        default=1000,
                        type=int)  # Number of episodes to train for
    parser.add_argument("--steps_per_epoch",
                        default=1000,
                        type=int)  # number of batches trained on per epoch call
    parser.add_argument("--batch_size",
                        default=4,
                        type=int)  # Batch size for training
    parser.add_argument("--image_width",
                        default=512,
                        type=int,
                        help="Your network wants to divide it to 2 for a while.\\"
                             "So a tip is to find the aspect ration you need and mulitply until you have a sufficient width\\"
                             "example: So 4*2*2*2*2*2*2*2 = 512, 3*2*2*2*2*2*2*2=384")  # image width
    parser.add_argument("--image_height",
                        default=384,
                        type=int,
                        help="Your network wants to divide it to 2 for a while.\\"
                             "So a tip is to find the aspect ration you need and mulitply until you have a sufficient height\\"
                             "example: So 4*2*2*2*2*2*2*2 = 512, 3*2*2*2*2*2*2*2=384")  # image height
    parser.add_argument("--image_channels",
                        default=3,
                        type=int,
                        help="1= grayscale, 3=rgb, 4=rgba")  # image channels
    parser.add_argument("--lr",
                        default=2e-5,
                        type=float)  # Learning rate
    parser.add_argument("--weight_decay",
                        default=1e-6)  # weight_decay
    parser.add_argument("--load_model_path",
                        default="")  # If should load model: if "" don't load anything
    parser.add_argument("--eval",
                        action='store_true')  # If we only want to evaluate a model.
    parser.add_argument("--cpu",
                        action='store_true')  # If we only want to evaluate a model.
    args = parser.parse_args()

    # Hard-coded values
    load_model = True
    categories = ["water"]
    nclasses = len(categories)
    validation_split = 0.2

    input_shape = (args.batch_size, args.image_height, args.image_width, args.image_channels)
    target_size = (args.image_height, args.image_width)

    as_gray = args.image_channels == 1
    model_name = 'unet' + "_" + str(args.image_width) + "_" + str(args.image_height) + '_binary.hdf5'

    # DATA PATH
    current_dir = os.getcwd()

    data_folder = Path(current_dir, "..", "Segmentation_Data")
    train_folder = Path(data_folder, "train")
    test_folder = Path(data_folder, "test")

    tensorboard_path = Path(data_folder, "tensorboard")

    result_folder = Path(data_folder, "result", model_name.split(".")[0])
    if result_folder.exists():
        rm_tree(result_folder)
    result_folder.mkdir(parents=True)

    checkpoint_dir = Path(current_dir, "saved_model")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    # Choose an optimizer and loss function for training:
    # loss_object = tf.keras.losses.BinaryCrossentropy()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr)

    # Create model:
    model = UNetBinary()
    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
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

    if args.cpu:
        with tf.device('/cpu:0'):
            if not args.eval:  # Training
                model = train_model(model=model,
                                    target_size=target_size,
                                    batch_size=args.batch_size,
                                    list_of_categories=categories,
                                    train_epoch=args.epochs,
                                    steps_per_epoch=args.steps_per_epoch,
                                    validation_split=validation_split,
                                    train_folder=train_folder,
                                    checkpoint_path=checkpoint_path,
                                    tensorboard_path=tensorboard_path)
            # else just eval
            test_model(model=model,
                       target_size=target_size,
                       batch_size=args.batch_size,
                       list_of_categories=categories,
                       as_gray=as_gray,
                       test_folder=test_folder,
                       result_folder=result_folder)
    else:
        if not args.eval:  # Training
            model = train_model(model=model,
                                target_size=target_size,
                                batch_size=args.batch_size,
                                list_of_categories=categories,
                                train_epoch=args.epochs,
                                steps_per_epoch=args.steps_per_epoch,
                                validation_split=validation_split,
                                train_folder=train_folder,
                                checkpoint_path=checkpoint_path,
                                tensorboard_path=tensorboard_path)
        # else just eval
        test_model(model=model,
                   target_size=target_size,
                   batch_size=args.batch_size,
                   list_of_categories=categories,
                   as_gray=as_gray,
                   test_folder=test_folder,
                   result_folder=result_folder)
