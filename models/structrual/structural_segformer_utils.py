from keras.layers import *
import tensorflow as tf


def convolutional_block(inputs, n_filters, kernel_size, strides=1, norm=True, activation=None):
    x = Conv2D(n_filters, kernel_size, strides, padding='same')(inputs)
    x = BatchNormalization(momentum=.99, epsilon=1e-5)(x) if norm is True else x
    x = Activation(activation)(x) if activation else x
    return x


def decoder_mlp(inputs, embed_dim):
    """
    input shape -> [batches, height, width, embed_dim]
    :return:  shape -> [batches, n_patches, embed_dim]
    """
    batches, height, width, channels = inputs.shape
    x = tf.reshape(inputs, shape=[-1, height * width, channels])
    x = Dense(embed_dim, use_bias=True)(x)

    return x


def seg_former_decoder_block(inputs, embed_dim, up_size=(4, 4)):
    """
    inputs: shape -> [batches, height, width, embed_dim]
    :return: shape -> [batches, height, width, embed_dim]
    """
    batches, height, width, channels = inputs.shape
    x = decoder_mlp(inputs, embed_dim)
    x = tf.reshape(x, shape=[-1, height, width, embed_dim])
    x = UpSampling2D(size=up_size, interpolation='bilinear')(x)

    return x


def seg_former_head(features, embed_dim, num_classes, drop_rate=0.):
    assert len(features) == 4
    assert len(set(feature.shape for feature in features)) == 1

    x = Concatenate(axis=-1)(features)
    x = convolutional_block(x, n_filters=embed_dim, kernel_size=1, norm=True, activation='relu')
    x = Dropout(rate=drop_rate)(x)
    x = Conv2D(num_classes, kernel_size=1, activation='softmax')(x)
    return x
