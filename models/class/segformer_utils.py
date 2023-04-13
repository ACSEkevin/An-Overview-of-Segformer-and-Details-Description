from keras.layers import *
from keras.initializers.initializers_v2 import LecunNormal, Zeros, GlorotUniform
import tensorflow as tf
from typing import Any


class ConvolutionalBlock(Layer):
    def __init__(self, n_filters, kernel_size=1, strides=1, padding: Any = 'valid',
                 batch_norm=True, activation='relu', drop_rate=0., ):
        super(ConvolutionalBlock, self).__init__()
        self.down_sample = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.batch_norm = BatchNormalization(momentum=.99, epsilon=1e-6) \
            if batch_norm is True else Activation('linear')
        self.activation = Activation(activation) if activation else Activation('linear')
        self.dropout = Dropout(rate=drop_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.activation(self.batch_norm(self.down_sample(inputs)))
        return self.dropout(x)


class MLP(Layer):
    """
    inputs.shape -> [batches, height, width, embed_dim]
    outputs.shape -> [batches, height * width, embed_dim]
    """
    def __init__(self, embed_dim=768):
        super(MLP, self).__init__()
        self.mlp = Dense(embed_dim, use_bias=True, kernel_initializer=GlorotUniform(),
                         bias_initializer=Zeros())

    def call(self, inputs, *args, **kwargs):
        batches, height, width, embed_dim = inputs.shape
        x = tf.reshape(inputs, shape=[batches, -1, embed_dim])
        x = self.mlp(x)
        return x


class SegFormerDecoder(Layer):
    """
    inputs from SegFormer backbones, length == 4, shape -> [batches, height_patch, width_patch, embed_dim]
    :return
    """
    def __init__(self, num_classes, version='B0', interpolation='bilinear', drop_rate=0.,):
        super(SegFormerDecoder, self).__init__()
        self.up_sampling_scalers = [(4, 4), (8, 8), (16, 16), (32, 32)]

        embed_dim = 256 if version in ['B0', 'B1'] else 768

        self.feature_linear_list = [MLP(embed_dim=embed_dim) for _ in range(4)]
        self.feature_up_list = [UpSampling2D(
            size=self.up_sampling_scalers[index], interpolation=interpolation
        ) for index in range(len(self.up_sampling_scalers))]

        self.feature_concat = Concatenate(axis=-1)
        self.dropout = Dropout(rate=drop_rate)
        self.linear_fusion = ConvolutionalBlock(embed_dim, kernel_size=1, activation='relu')
        self.outputs = Conv2D(num_classes, kernel_size=1, activation='softmax')

    def call(self, inputs, *args, **kwargs):
        assert isinstance(inputs, tuple) or isinstance(inputs, list)
        assert len(inputs) == 4
        batches, height, width, embed_dim = inputs[0].shape

        features_decode = list()
        for index in range(len(inputs)):
            x = self.feature_linear_list[index](inputs[index])
            x = tf.reshape(x, shape=[batches, inputs[index].shape[1], inputs[index].shape[2], -1])
            x = self.feature_up_list[index](x)
            features_decode.append(x)

        x = self.feature_concat(features_decode)
        x = self.linear_fusion(x)
        x = self.dropout(x)
        x = self.outputs(x)

        return x







