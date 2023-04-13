from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Conv2DTranspose, UpSampling2D
from keras.layers import Input, BatchNormalization, Add, Concatenate, Dropout, Layer
from keras.models import Sequential, Model

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'},
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3,
        'mode': 'fan_out',
        'distribution': 'uniform'},
}


class DoubleConvBlock(Layer):
    def __init__(self, n_filters):
        super(DoubleConvBlock, self).__init__()
        self.n_filters = n_filters
        self.feature_conv0 = Conv2D(n_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm0 = BatchNormalization()
        self.activation0 = Activation('relu')

        self.feature_conv1 = Conv2D(n_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm1 = BatchNormalization()
        self.activation1 = Activation('relu')

    def call(self, inputs, *args, **kwargs):
        x = self.feature_conv0(inputs)
        x = self.batch_norm0(x)
        x = self.activation0(x)

        x = self.feature_conv1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_filters': self.n_filters
        })


class DownSamplingBlock(Layer):
    def __init__(self, pool_stride=(2, 2), drop_rate=0.):
        super(DownSamplingBlock, self).__init__()
        assert pool_stride == (2, 2)
        self.pool_stride = pool_stride
        self.drop_rate = drop_rate
        self.down_sampling = MaxPooling2D(pool_size=(2, 2), strides=pool_stride)
        self.drop_out = Dropout(rate=drop_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.down_sampling(inputs)
        if self.drop_rate > 0.:
            x = self.drop_out(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_stride': self.pool_stride,
            'drop_rate': self.drop_rate
        })


class UpSamplingBlock(Layer):
    def __init__(self, n_filters, drop_rate=0., interpolate=None):
        super(UpSamplingBlock, self).__init__()
        assert interpolate is None
        self.n_filters = n_filters
        self.drop_rate = drop_rate
        self.interpolate = interpolate
        if interpolate is None:
            self.up_sampling = Conv2DTranspose(n_filters, kernel_size=(3, 3), strides=2, padding='same')
        else:
            self.up_sampling = UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.feature_concat = Concatenate()
        self.drop_out = Dropout(rate=drop_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.up_sampling(inputs[0])
        x = self.feature_concat([x, inputs[1]])
        if self.drop_rate > 0.:
            x = self.drop_out(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_filters': self.n_filters,
            'drop_rate': self.drop_rate,
            'interpolate': self.interpolate
        })


def UNet(input_shape, classes, pool_drop_rate=0., up_drop_rate=0.):
    """

    :param up_drop_rate:
    :param pool_drop_rate:
    :param input_shape:
    :param classes:
    :return:
    """
    ins = Input(input_shape)

    # encoder architecture
    # [224, 224, 3] -> [112, 112, 64]
    conv1 = DoubleConvBlock(64)(ins)
    pool1 = DownSamplingBlock(drop_rate=pool_drop_rate)(conv1)
    # [112, 112, 64] -> [56, 56, 128]
    conv2 = DoubleConvBlock(128)(pool1)
    pool2 = DownSamplingBlock(drop_rate=pool_drop_rate)(conv2)
    # [56, 56, 128] -> [28, 28, 256]
    conv3 = DoubleConvBlock(256)(pool2)
    pool3 = DownSamplingBlock(drop_rate=pool_drop_rate)(conv3)
    # [28, 28, 256] -> [14, 14, 512]
    conv4 = DoubleConvBlock(512)(pool3)
    pool4 = DownSamplingBlock(drop_rate=pool_drop_rate)(conv4)
    # [14, 14, 512] -> [7, 7, 1024]
    bottle_neck = DoubleConvBlock(1024)(pool4)

    # decoder architecture
    tran1 = UpSamplingBlock(512, drop_rate=up_drop_rate)([bottle_neck, conv4])
    conv5 = DoubleConvBlock(512)(tran1)

    tran2 = UpSamplingBlock(256, drop_rate=up_drop_rate)([conv5, conv3])
    conv6 = DoubleConvBlock(256)(tran2)

    tran3 = UpSamplingBlock(128, drop_rate=up_drop_rate)([conv6, conv2])
    conv7 = DoubleConvBlock(128)(tran3)

    tran4 = UpSamplingBlock(64, drop_rate=up_drop_rate)([conv7, conv1])
    conv8 = DoubleConvBlock(64)(tran4)

    outs = Conv2D(classes, kernel_size=(1, 1), padding='same', activation='softmax')(conv8)

    model = Model(ins, outs, name='UNet')

    return model


