## https://github.com/ErikStammes/SemanticSegmentation/blob/master/model.py
## https://www.kaggle.com/advaitsave/tensorflow-2-nuclei-segmentation-unet/notebook
## https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model


class UNet(Model):
    def __init__(self, nclasses=1):
        super(UNet, self).__init__()

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

        # Build U-Net model
        # self.prepare = Lambda(lambda x: x / 255)
        self.prepare = Lambda(lambda x: x / 255)

        self.block1 = Sequential([
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.1),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
        ])
        # Down
        self.block2 = Sequential([
            MaxPooling2D((2, 2)),  # Is from 1. u-block
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.1),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization()
        ])
        # Down
        self.block3 = Sequential([
            MaxPooling2D((2, 2)),  # Is from 2. u-block
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization()
        ])
        # Down
        self.block4 = Sequential([
            MaxPooling2D((2, 2)),  # Is from 3. u-block
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization()
        ])

        # bottom
        self.block5 = Sequential([
            MaxPooling2D(pool_size=(2, 2)),  # Is from 4. u-block
            Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
        ])

        self.block6 = Sequential([
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        ])

        self.block7 = Sequential([
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        ])

        self.block8 = Sequential([
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        ])

        self.block9 = Sequential([
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.1),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2D(nclasses, (1, 1), activation='softmax', dtype=tf.float32)
        ])

    def call(self, inputs, training=False, **kwargs):
        # Preparing input:
        # prep_inputs = self.prepare(inputs)
        # Going down the U
        inputs = self.prepare(inputs)

        block1 = self.block1(inputs)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)  # Bottom

        # Going up the U
        concat = concatenate([block4, block5])
        block6 = self.block6(concat)
        concat = concatenate([block3, block6])
        block7 = self.block7(concat)
        concat = concatenate([block2, block7])
        block8 = self.block8(concat)
        concat = concatenate([block1, block8])
        output = self.block9(concat)

        # block6 = self.block6(block5)
        # block7 = self.block7(block6)
        # block8 = self.block8(block7)
        # output = self.block9(block8)
        return output


class UNetBinary(Model):
    def __init__(self):
        super(UNetBinary, self).__init__()
        # Build U-Net model

        self.prepare = Lambda(lambda x: x / 255)

        self.block1 = Sequential([
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.1),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
        ])
        # Down
        self.block2 = Sequential([
            MaxPooling2D((2, 2)),  # Is from 1. u-block
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.1),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization()
        ])
        # Down
        self.block3 = Sequential([
            MaxPooling2D((2, 2)),  # Is from 2. u-block
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization()
        ])
        # Down
        self.block4 = Sequential([
            MaxPooling2D((2, 2)),  # Is from 3. u-block
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization()
        ])

        # bottom
        self.block5 = Sequential([
            MaxPooling2D(pool_size=(2, 2)),  # Is from 4. u-block
            Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
        ])

        self.block6 = Sequential([
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        ])

        self.block7 = Sequential([
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        ])

        self.block8 = Sequential([
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        ])

        self.block9 = Sequential([
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.1),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            # Conv2D(2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            Conv2D(1, (1, 1), activation='sigmoid')
        ])

    def call(self, inputs, training=False, **kwargs):
        # Preparing input:
        # prep_inputs = self.prepare(inputs)
        inputs = self.prepare(inputs)

        # Going down the U
        block1 = self.block1(inputs)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)  # Bottom

        # Going up the U
        concat = concatenate([block4, block5])
        block6 = self.block6(concat)
        concat = concatenate([block3, block6])
        block7 = self.block7(concat)
        concat = concatenate([block2, block7])
        block8 = self.block8(concat)
        concat = concatenate([block1, block8])
        output = self.block9(concat)
        return output


class UNetSmall(Model):
    def __init__(self, nclasses=1):
        super(UNetSmall, self).__init__()

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

        # Build U-Net model
        # self.prepare = Lambda(lambda x: x / 255)
        self.prepare = Lambda(lambda x: x / 255)

        self.down1 = Sequential([
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.1),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
        ], name="down1")
        # Down
        self.down2 = Sequential([
            MaxPooling2D((2, 2)),  # Is from 1. u-block
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.1),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization()
        ], name="down2")

        # bottom
        self.down3 = Sequential([
            MaxPooling2D(pool_size=(2, 2)),  # Is from 2. u-block
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
        ], name='down3')

        self.bottom = Sequential([
            MaxPooling2D((2, 2)),  # Is from 3. u-block
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        ], name='bottom')

        self.up1 = Sequential([
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        ], name='up1')

        self.up2 = Sequential([
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        ], name='up2')

        self.up3 = Sequential([
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Dropout(0.1),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
            BatchNormalization(),
            Conv2D(nclasses, (1, 1), activation='softmax', dtype=tf.float32)
        ], name='up3')

    def call(self, inputs, training=False, **kwargs):
        # Preparing input:
        # prep_inputs = self.prepare(inputs)
        # Going down the U
        inputs = self.prepare(inputs)

        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        bottom = self.bottom(down3)
        concat = concatenate([bottom, down3])
        up1 = self.up1(concat)
        concat = concatenate([up1, down2])
        up2 = self.up2(concat)
        concat = concatenate([up2, down1])
        up3 = self.up3(concat)
        return up3
