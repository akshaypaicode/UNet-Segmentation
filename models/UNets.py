# -*- coding: utf-8 -*-
"""
Created on Sat May 12 03:42:42 2018

@author: akshay
"""

# Import keras modules
import keras.backend as K
from keras.models import Model
from keras import regularizers
from keras.layers import Input, BatchNormalization, Cropping2D
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
import numpy as np

# Set tf dim ordering
K.set_image_dim_ordering("tf")


class UNet(Model):
    def __init__(self, img_rows, img_cols, n_classes, out_activation="softmax",
                 complexity_factor=1, l1_reg=None, l2_reg=None,
                 base_model=None, logger=None, **kwargs):

        self.img_shape = (img_rows, img_cols, 1)
        self.n_classes = n_classes
        self.cf = np.sqrt(complexity_factor)

        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        # Build model and init base keras Model class
        if not base_model:
            # New training session
            Model.__init__(self, *self.init_model(out_activation, l1_reg,
                                                  l2_reg, **kwargs))
        else:
            # Resumed training
            Model.__init__(self, base_model.input, base_model.output)

    def init_model(self, out_activation, l1, l2, **kwargs):
        """
        Build the UNet model with the specified input image shape.

        OBS: Depending on image dim cropping may be necessary between layers

        OBS: In some cases, the output is smaller than the input.
        self.label_crop stores the number of pixels that must be cropped from
        the target labels matrix to compare correctly.
        """
        inputs = Input(shape=self.img_shape)

        # Apply regularization if not None or 0
        # l1 regularization used for layer activity
        # l2 regularization used for convolution weights
        if l2:
            kr = regularizers.l2(l2)
        else:
            kr = None
        if l1:
            ar = regularizers.l1(l1)
        else:
            ar = None

        """
        Contracting path
        Note: Listed tensor shapes assume img_row = 256, img_col = 256
        """
        # [256, 256, 1] -> [256, 256, 64] -> [256, 256, 64] -> [128, 128, 64]
        conv1 = Conv2D(int(64*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(inputs)
        conv1 = Conv2D(int(64*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv1)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

        # [128, 128, 64] -> [128, 128, 128] -> [128, 128, 128] -> [64, 64, 128]
        conv2 = Conv2D(int(128*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(pool1)
        conv2 = Conv2D(int(128*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv2)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

        # [64, 64, 128] -> [64, 64, 256] -> [64, 64, 256] -> [32, 32, 256]
        conv3 = Conv2D(int(256*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(pool2)
        conv3 = Conv2D(int(256*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv3)
        bn3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

        # [32, 32, 256] -> [32, 32, 512] -> [32, 32, 512] -> [16, 16, 512]
        conv4 = Conv2D(int(512*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(pool3)
        conv4 = Conv2D(int(512*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv4)
        bn4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

        # [16, 16, 512] -> [16, 16, 1024] -> [16, 16, 1024]
        conv5 = Conv2D(int(1024*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(pool4)
        conv5 = Conv2D(int(1024*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv5)
        bn5 = BatchNormalization()(conv5)

        """
        Up-sampling
        """
        # [16, 16, 1024] -> [32, 32, 1024] -> [32, 32, 512]
        up1 = UpSampling2D(size=(2, 2))(bn5)
        conv6 = Conv2D(int(512*self.cf), 2, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(up1)
        bn6 = BatchNormalization()(conv6)

        # Merge conv4 [32, 32, 512] with conv6 [32, 32, 512]
        # --> [32, 32, 1024]
        cropped_bn4 = self.crop_nodes_to_match(bn4, bn6)
        merge6 = Concatenate(axis=-1)([cropped_bn4, bn6])

        # [32, 32, 1024] -> [32, 32, 512] -> [32, 32, 512]
        conv6 = Conv2D(int(512*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(merge6)
        conv6 = Conv2D(int(512*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv6)
        bn7 = BatchNormalization()(conv6)

        # [32, 32, 512] -> [64, 64, 512] -> [64, 64, 256]
        up2 = UpSampling2D(size=(2, 2))(bn7)
        conv7 = Conv2D(int(256*self.cf), 2, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(up2)
        bn8 = BatchNormalization()(conv7)

        # Merge conv3 [64, 64, 256] with conv7 [64, 64, 256]
        # --> [32, 32, 512]
        cropped_bn3 = self.crop_nodes_to_match(bn3, bn8)
        merge7 = Concatenate(axis=-1)([cropped_bn3, bn8])

        # [64, 64, 512] -> [64, 64, 256] -> [64, 64, 256]
        conv7 = Conv2D(int(256*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(merge7)
        conv7 = Conv2D(int(256*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv7)
        bn9 = BatchNormalization()(conv7)

        # [64, 64, 256] -> [128, 128, 256] -> [128, 128, 128]
        up3 = UpSampling2D(size=(2, 2))(bn9)
        conv8 = Conv2D(int(128*self.cf), 2, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(up3)
        bn10 = BatchNormalization()(conv8)

        # Merge conv2 [128, 128, 128] with conv8 [128, 128, 128]
        # --> [128, 128, 256]
        cropped_bn2 = self.crop_nodes_to_match(bn2, bn10)
        merge8 = Concatenate(axis=-1)([cropped_bn2, bn10])

        # [128, 128, 256] -> [128, 128, 128] -> [128, 128, 128]
        conv8 = Conv2D(int(128*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(merge8)
        conv8 = Conv2D(int(128*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv8)
        bn11 = BatchNormalization()(conv8)

        # [128, 128, 128] -> [256, 256, 128] -> [256, 256, 64]
        up4 = UpSampling2D(size=(2, 2))(bn11)
        conv9 = Conv2D(int(64*self.cf), 2, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(up4)
        bn12 = BatchNormalization()(conv9)

        # Merge conv1 [256, 256, 64] with conv9 [256, 256, 64]
        # --> [256, 256, 128]
        cropped_bn1 = self.crop_nodes_to_match(bn1, bn12)
        merge9 = Concatenate(axis=-1)([cropped_bn1, bn12])

        # [256, 256, 128] -> [256, 256, 64] -> [256, 256, 64]
        conv9 = Conv2D(int(64*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(merge9)
        conv9 = Conv2D(int(64*self.cf), 3, activation='relu', padding='same',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv9)
        bn13 = BatchNormalization()(conv9)

        """
        Output modeling layer
        """
        # [256, 256, 64] -> [256, 256, n_classes]
        out = Conv2D(self.n_classes, 1, activation=out_activation)(bn13)

        return [inputs], [out]

    def crop_nodes_to_match(self, node1, node2):

        s1 = np.array(node1.get_shape().as_list())[1:-1]
        s2 = np.array(node2.get_shape().as_list())[1:-1]

        if np.any(s1 != s2):
            c = (s1 - s2).astype(np.int)
            cr = np.array([c//2, c//2]).T
            cr[:, 1] += c % 2
            cropped_node1 = Cropping2D(cr)(node1)
            self.label_crop += cr
        else:
            cropped_node1 = node1

        return cropped_node1
