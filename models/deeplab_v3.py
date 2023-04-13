"""
This DeepLab V3 is referenced from Pytorch Official DeepLabV3 model
<https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation>
"""

from keras.layers import *
from keras.models import Model


class ConvBatchNormReLU(Layer):
    def __init__(self, n_filters, kernel_size, strides=1, dilation_rate=1,
                 padding='same', norm=True, activation='relu'):
        super(ConvBatchNormReLU, self).__init__()
        self.conv = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding=padding,
                           dilation_rate=dilation_rate)
        self.batch_norm = BatchNormalization(momentum=.99, epsilon=1e-4) \
            if norm is True else Activation('linear')
        self.activation = Activation(activation) if activation else Activation('linear')

    def call(self, inputs, *args, **kwargs):
        return self.activation(self.batch_norm(self.conv(inputs)))


class ConvolutionalBlock(Layer):
    def __init__(self, filters: tuple or list, kernel_size, strides, dilation_rate=1):
        super(ConvolutionalBlock, self).__init__()
        assert isinstance(filters, tuple) or isinstance(filters, list)
        assert len(filters) == 3

        self.main_conv0 = ConvBatchNormReLU(filters[0], kernel_size=1, strides=strides, padding='valid')
        self.main_conv1 = ConvBatchNormReLU(filters[1], kernel_size=kernel_size, strides=1,
                                            dilation_rate=dilation_rate)
        self.main_conv2 = ConvBatchNormReLU(filters[2], kernel_size=1, activation='linear')

        # FIXME <strides>
        self.shortcut = ConvBatchNormReLU(filters[2], kernel_size=1, strides=1,
                                          padding='valid', activation='linear')
        self.feature_add = Add()
        self.activation = Activation('relu')

    def call(self, inputs, *args, **kwargs):
        x = self.main_conv0(inputs)
        x = self.main_conv1(x)
        x = self.main_conv2(x)

        shortcut = self.shortcut(inputs)

        x = self.feature_add([x, shortcut])
        x = self.activation(x)
        return x


class IdentityBlock(Layer):
    def __init__(self, filters: tuple or list, kernel_size, dilation_rate=1):
        super(IdentityBlock, self).__init__()
        assert isinstance(filters, tuple) or isinstance(filters, list)
        assert len(filters) == 3

        self.main_conv0 = ConvBatchNormReLU(filters[0], kernel_size=1, strides=1, padding='valid')
        self.main_conv1 = ConvBatchNormReLU(filters[1], kernel_size=kernel_size, strides=1,
                                            dilation_rate=dilation_rate)
        self.main_conv2 = ConvBatchNormReLU(filters[2], kernel_size=1, activation='linear')
        self.feature_add = Add()
        self.activation = Activation('relu')

    def call(self, inputs, *args, **kwargs):
        x = self.main_conv0(inputs)
        x = self.main_conv1(x)
        x = self.main_conv2(x)

        x = self.feature_add([x, inputs])
        x = self.activation(x)
        return x


def _make_layer(n_blocks, filters, kernel_size=3, strides=1, conv_dil_rate: int = 1, id_dil_rate: int = 1):
    blocks = [ConvolutionalBlock(filters, kernel_size, strides, conv_dil_rate)]
    for _ in range(1, n_blocks):
        blocks.append(IdentityBlock(filters, kernel_size, id_dil_rate))

    return blocks


class DeepLabBackBone(Layer):
    def __init__(self, version='50', zero_padding=False, aux=False):
        super(DeepLabBackBone, self).__init__()
        filters = {'50': [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)],
                   '101': [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)],
                   '152': [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]}

        blocks = {'50': [3, 4, 6, 3], '101': [3, 4, 23, 3], '152': [3, 8, 36, 3]}
        assert version in filters.keys()

        self.filter_list = filters[version]
        self.block_list = blocks[version]
        self.auxiliary = aux
        self.dilation_rates = [(1, 1), (1, 1), (1, 2), (2, 4)]

        self.zero_padding = ZeroPadding2D(padding=(3, 3)) if zero_padding is True else Activation('linear')
        self.down_sample = ConvBatchNormReLU(64, kernel_size=7, strides=2, padding='same')
        self.max_pool = MaxPooling2D(pool_size=(2, 2))

        self.stages = [
            _make_layer(self.block_list[index], self.filter_list[index], strides=2 if index == 1 else 1,
                        conv_dil_rate=self.dilation_rates[index][0],
                        id_dil_rate=self.dilation_rates[index][-1]) for index in range(len(self.block_list))
        ]

    def call(self, inputs, *args, **kwargs):
        aux_feature = None
        x = self.zero_padding(inputs)
        x = self.down_sample(x)
        # print(x.shape)
        x = self.max_pool(x)
        # print(x.shape)

        for index, block_list in enumerate(self.stages):
            for block in block_list:
                x = block(x)
            if index == 2 and self.auxiliary is True:
                aux_feature = x
            # print(x.shape)

        print('Backbone auxiliary: {}'.format(self.auxiliary))
        # return x, aux_feature if self.auxiliary is True else x
        return x


class ASPPAdaptivePooling(Layer):
    def __init__(self):
        super(ASPPAdaptivePooling, self).__init__()
        self.size = (60, 60)

        self.adaptive_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.dim_reduction = ConvBatchNormReLU(256, kernel_size=1)
        self.interpolate = UpSampling2D(size=self.size, interpolation='bilinear')

    def call(self, inputs, *args, **kwargs):
        batches, height, width, channels = inputs.shape
        self.interpolate.size = (height, width)
        x = self.adaptive_avg_pool(inputs)
        x = self.dim_reduction(x)
        x = self.interpolate(x)
        # x = UpSampling2D(size=(height, width), interpolation='bilinear')(x)

        assert x.shape[1] == height and x.shape[2] == width, \
            "received height {}, width {}, expected size {}".format(height, width, self.size)
        return x


class AtrousSpatialPyramidPooling(Layer):
    def __init__(self, dilation_rates=None, drop_rate=0.5):
        super(AtrousSpatialPyramidPooling, self).__init__()
        assert dilation_rates is None or len(dilation_rates) == 4
        self.dilation_rates = [1, 12, 24, 36] if not dilation_rates else dilation_rates
        self.branches = [ConvBatchNormReLU(
            256, kernel_size=1 if index == 0 else 3, dilation_rate=self.dilation_rates[index], padding='same'
        ) for index in range(len(self.dilation_rates))]

        self.branches.append(ASPPAdaptivePooling())
        self.feature_concat = Concatenate(axis=-1)
        self.dim_reduction = ConvBatchNormReLU(n_filters=256, kernel_size=1, strides=1, padding='same')
        self.dropout = Dropout(rate=drop_rate)

    def call(self, inputs, *args, **kwargs):
        features = [branch(inputs) for branch in self.branches]
        x = self.feature_concat(features)
        x = self.dim_reduction(x)
        x = self.dropout(x)

        assert len(features) == 5, 'expected length: 5, got: {}'.format(len(features))
        assert x.shape[1] == inputs.shape[1] and x.shape[2] == inputs.shape[2]
        return x


class DeepLabHead(Layer):
    def __init__(self, num_classes, up_size=(8, 8), aspp_drop_rate=0.5):
        super(DeepLabHead, self).__init__()
        self.aspp_module = AtrousSpatialPyramidPooling(drop_rate=aspp_drop_rate)
        self.down_sample = ConvBatchNormReLU(256, kernel_size=3, strides=1, padding='same')
        self.projection = Conv2D(num_classes, kernel_size=1, strides=1, padding='same')
        self.interpolate = UpSampling2D(size=up_size, interpolation='bilinear')
        self.softmax = Softmax(axis=-1)

    def call(self, inputs, *args, **kwargs):
        x = self.aspp_module(inputs)
        x = self.down_sample(x)
        x = self.projection(x)
        x = self.softmax(self.interpolate(x))

        return x


class AuxiliaryFCNHead(Layer):
    """
    Auxiliary head should be branched from backbone stage 3
    """

    def __init__(self, num_classes, up_size=(8, 8), drop_rate=0.):
        super(AuxiliaryFCNHead, self).__init__()
        self.down_sample = ConvBatchNormReLU(256, kernel_size=3, strides=1, padding='same')
        self.drop_out = Dropout(rate=drop_rate)
        self.projection = Conv2D(num_classes, kernel_size=1, strides=1, padding='same')
        self.interpolate = UpSampling2D(size=up_size, interpolation='bilinear')
        self.softmax = Softmax(axis=-1)

    def call(self, inputs, *args, **kwargs):
        x = self.down_sample(inputs)
        x = self.drop_out(x)
        x = self.projection(x)
        x = self.softmax(self.interpolate(x))

        assert inputs.shape[1:] == (60, 60, 1024)
        return x


def DeepLabV3(input_shape, num_classes, backbone='resnet50', up_size=(8, 8),
              aspp_drop_rate=0., aux=False, aux_drop=0.):
    """
    This model DeepLab V3 is referenced from Pytorch Official DeepLabV3 model
    """
    backbones = {'resnet50': DeepLabBackBone(version='50', zero_padding=False, aux=aux),
                 'resnet101': DeepLabBackBone(version='101', zero_padding=False, aux=aux),
                 'resnet152': DeepLabBackBone(version='152', zero_padding=False, aux=aux)}

    inputs = Input(input_shape)

    if aux is True:
        x, aux_feature = backbones[backbone](inputs)
        aux_out = AuxiliaryFCNHead(num_classes, up_size=up_size, drop_rate=aux_drop)(aux_feature)
    else:
        x = backbones[backbone](inputs)

    # print("model shape after backbone", x.shape)
    x = DeepLabHead(num_classes, up_size=up_size, aspp_drop_rate=aspp_drop_rate)(x)

    if aux is True:
        model = Model(inputs, [x, aux_out], name='deep_lab_v3')
    else:
        model = Model(inputs, x, name='deep_lab_v3')

    assert x.shape[1] == input_shape[0] and x.shape[2] == input_shape[1]
    return model
