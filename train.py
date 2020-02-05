from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

'''-----------------------------------------
Data Iteration
-----------------------------------------'''
def adjustData(img):
    img = preprocess_input(img)
    return img
################################################################################
def data_augmentation ():
    data_gen_args = dict()

    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        fill_mode='nearest')

    return data_gen_args
################################################################################
def data_generator (batch_size, image_npy, label_npy):
    aug_dict = data_augmentation ()
    image_data = np.load(image_npy)
    label_data = np.load(label_npy)
    new_label_data = []
    h, w = label_data[0].shape
    for label in label_data:
        label = np.reshape(label, (h, w, 1))
        new_label_data.append(label)
    label_data = np.array(new_label_data)
    print (image_data.shape, label_data.shape)
    datagen_img = ImageDataGenerator(**aug_dict)
    datagen_msk = ImageDataGenerator(**aug_dict)
    img_gen = datagen_img.flow(image_data, batch_size=batch_size, seed = 1)
    msk_gen = datagen_msk.flow(label_data, batch_size=batch_size, seed = 1)
    data_generator = zip (img_gen, msk_gen)
    for img, mask in data_generator:
        img = adjustData(img)
        yield (img, mask)
################################################################################
'''Load Libraries'''
################################################################################
from keras.models import Model
import keras
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, Reshape, Permute, BatchNormalization, GlobalAveragePooling2D, Dropout,Dense
from keras.layers.convolutional import Conv2D
from keras.layers import UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
import tensorflow as tf
import numpy as np
import time
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from keras.layers.merge import Concatenate
import argparse
from keras import optimizers
################################################################################
################################################################################
'''--------------------------------------
Feature Pyramid structure (FPN)
--------------------------------------'''
class UpsampleLike(keras.layers.Layer):
    """
    Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = tf.transpose(source, (0, 2, 3, 1))
            output = tf.image.resize_images(source, (target_shape[2], target_shape[3]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = tf.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return tf.image.resize_images(source, (target_shape[1], target_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
################################################################################
def FPN (C2, C3, C4, C5, feature_size=256):
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3_upsampled = UpsampleLike(name='P3_upsampled')([P3, C2])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)
    P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P2 = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
    P2 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)
    return P2, P3, P4, P5
################################################################################
def classifier_resNet (input):
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
        weights='imagenet',input_tensor=(input),input_shape=(None,None,3))
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(1, activation= 'sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    return model
###################################################################
def get_model():
    img_input_shape = (None, None, 3)
    img_input = Input(shape=img_input_shape)
    model = classifier_resNet(img_input)
    return model
################################################################################
def FPN_backbone ():
    img_input_shape = (None, None, 3)
    img_input = Input(shape=img_input_shape)
    model = classifier_resNet(img_input)

    C2=(model.layers[37]).output
    C3=(model.layers[79]).output
    C4=(model.layers[141]).output
    C5=(model.layers[173]).output

    P2, P3, P4, P5 = FPN (C2, C3, C4, C5, feature_size=256)
    return img_input, P2, P3, P4, P5

'''--------------------------------------
Model for Segmentation (classifier backbone)
--------------------------------------'''

def relu(x): return Activation('relu')(x)
def sigmoid(x): return Activation('sigmoid')(x)
def linear(x): return Activation('linear')(x)

def conv(x, nf, ks, name, stride, padding, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), strides = stride, padding=padding, name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def upsampler (x, ks, name):
    up_x = UpSampling2D(size=(ks, ks), data_format=None , interpolation = 'bilinear', name=name)(x)
    return up_x

def D_features(P2, P3, P4, P5, branch, weight_decay):
    Df1 = conv(P2, 128, 3, "Df1_B%d" % branch, 1, 'same', (weight_decay, 0))
    Df2 = conv(P3, 128, 3, "Df2_B%d" % branch, 1, 'same', (weight_decay, 0))
    Df3 = conv(P4, 128, 3, "Df3_B%d" % branch, 1, 'same', (weight_decay, 0))
    Df4 = conv(P5, 128, 3, "Df4_B%d" % branch, 1, 'same', (weight_decay, 0))
    Ds1 = conv(Df1, 128, 3, "Ds1_B%d" % branch, 1, 'same', (weight_decay, 0))
    Ds2 = conv(Df2, 128, 3, "Ds2_B%d" % branch, 1, 'same', (weight_decay, 0))
    Ds3 = conv(Df3, 128, 3, "Ds3_B%d" % branch, 1, 'same', (weight_decay, 0))
    Ds4 = conv(Df4, 128, 3, "Ds4_B%d" % branch, 1, 'same', (weight_decay, 0))
    return Ds1, Ds2, Ds3, Ds4

def D_upsampled (Ds1, Ds2, Ds3, Ds4):
    Du1 = upsampler(Ds1, 1, "Du1")
    Du2 = upsampler(Ds2, 2, "Du2")
    Du3 = upsampler(Ds3, 4, "Du3")
    Du4 = upsampler(Ds4, 8, "Du4")

    return Du1, Du2, Du3, Du4

def concat (D1, D2, D3, D4):
    D = Concatenate()([D1, D2, D3, D4])
    return D

def forward (P2, P3, P4, P5, branch, classes, weight_decay):
    D1, D2, D3, D4 = D_features(P2, P3, P4, P5, branch, weight_decay)
    Du1, Du2, Du3, Du4 = D_upsampled(D1, D2, D3, D4)
    D = concat (Du1, Du2, Du3, Du4)
    redTocls = conv(D, 1, 3, "Dsm_seg", 1, 'same', (weight_decay, 0))
    upsampled = upsampler(redTocls, 4, "uped")
    flatten = conv(upsampled, 1, 1, "flat", 1, 'same', (weight_decay, 0))
    prediction = sigmoid(flatten)
    return prediction

def get_training_model(n_classes, layer):
    if layer == 1:
        img_input, C = one_lay ()
        flatten = conv(C, 1, 1, "flat", 1, 'same', (0, 0))
        output = linear(flatten)
        model = Model(inputs=img_input, outputs=output)
    else:
        img_input, p2 ,p3 ,p4 ,p5 = FPN_backbone()
        output = forward (p2, p3, p4, p5, 3, n_classes, 0)
        model = Model(inputs=img_input, outputs=output)
    return model

def get_testing_model(n_classes, layer):
    if layer == 1:
        img_input, C = one_lay ()
        flatten = conv(C, 1, 1, "flat", 1, 'same', (0, 0))
        output = linear(flatten)
        model = Model(inputs=img_input, outputs=output)
    else:
        img_input, p2 ,p3 ,p4 ,p5 = FPN_backbone()
        output = forward (p2, p3, p4, p5, 3, n_classes, 0)
        model = Model(inputs=img_input, outputs=output)
    return model

def eucl_loss(x, y):
    batch_size = 8
    l = K.sum(K.square(x - y)) / batch_size / 2
    return l

def eucl_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 10
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    print ("Learning rate is: {}".format(lr))
    return lr

'''----------------------------------
Training model - saves the best model
----------------------------------'''
def main(model_weights):
    train_gen = data_generator(4, 'Data/X_train.npy', 'Data/y_train.npy')
    print ("batch loaded")
    val_gen = data_generator(4, 'Data/X_valid.npy', 'Data/y_valid.npy')
    model = get_training_model(1, 4)
    model.summary()
    model_checkpoint = ModelCheckpoint(model_weights, monitor='val_loss',verbose=1, save_best_only=True)
    lr_checkpoint = LearningRateScheduler(lr_scheduler, verbose=0)

    start = time.time()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4),metrics = ['accuracy'])
    model.fit_generator(train_gen, epochs=30, steps_per_epoch=500, callbacks=[model_checkpoint, lr_checkpoint], validation_data=val_gen, validation_steps=50)
    print ("trained in {} seconds". format(time.time()-start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', help='model_name', required=True)
    args = parser.parse_args()
    model_weights = 'model_weight_' + str(args.n) + '.h5'
    main (model_weights)
