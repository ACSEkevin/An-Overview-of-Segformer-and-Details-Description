from structural_transformer_utils import *
from structural_segformer_utils import *
from keras.models import Model
import numpy as np
import tensorflow as tf


def SegFormer(input_shape,
              num_classes,
              n_blocks=None,
              embed_dims=None,
              decoder_embed_dim=256,
              patch_sizes=None,
              strides=None,
              heads=None,
              reduction_ratios=None,
              expansion_rate=None,
              attention_drop_rate=0.,
              drop_rate=0.,
              ):
    if expansion_rate is None:
        expansion_rate = [8, 8, 4, 4]
    if reduction_ratios is None:
        reduction_ratios = [8, 4, 2, 1]
    if heads is None:
        heads = [1, 2, 4, 8]
    if strides is None:
        strides = [4, 2, 2, 2]
    if patch_sizes is None:
        patch_sizes = [7, 3, 3, 3]
    if embed_dims is None:
        embed_dims = [32, 64, 160, 256]
    if n_blocks is None:
        n_blocks = [2, 2, 2, 2]

    block_range = np.cumsum([0] + n_blocks)
    attention_scheduler = np.linspace(0, attention_drop_rate, num=sum(n_blocks))
    projection_scheduler = np.linspace(0, drop_rate, num=sum(n_blocks))

    inputs = Input(input_shape)

    # encoder
    # stage 1
    x, height1, width1 = overlap_patch_embedding(inputs, embed_dims[0],
                                                 kernel_size=patch_sizes[0], strides=strides[0])

    for index in range(n_blocks[0]):
        attention_range = attention_scheduler[block_range[0]: block_range[1]]
        projection_range = projection_scheduler[block_range[0]: block_range[1]]
        x = seg_former_encoder_block(x, height1, width1, embed_dims[0], heads[0],
                                     sr_ratio=reduction_ratios[0],
                                     expansion_rate=expansion_rate[0],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature1 = tf.reshape(x, shape=[-1, height1, width1, embed_dims[0]])

    # stage 2
    x, height2, width2 = overlap_patch_embedding(feature1, embed_dims[1],
                                                 kernel_size=patch_sizes[1], strides=strides[1])

    for index in range(n_blocks[1]):
        attention_range = attention_scheduler[block_range[1]: block_range[2]]
        projection_range = projection_scheduler[block_range[1]: block_range[2]]
        x = seg_former_encoder_block(x, height2, width2, embed_dims[1], heads[1],
                                     sr_ratio=reduction_ratios[1],
                                     expansion_rate=expansion_rate[1],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature2 = tf.reshape(x, shape=[-1, height2, width2, embed_dims[1]])

    # stage 3
    x, height3, width3 = overlap_patch_embedding(feature2, embed_dims[2],
                                                 kernel_size=patch_sizes[2], strides=strides[2])
    for index in range(n_blocks[2]):
        attention_range = attention_scheduler[block_range[2]: block_range[3]]
        projection_range = projection_scheduler[block_range[2]: block_range[3]]
        x = seg_former_encoder_block(x, height3, width3, embed_dims[2], heads[2],
                                     sr_ratio=reduction_ratios[2],
                                     expansion_rate=expansion_rate[2],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature3 = tf.reshape(x, shape=[-1, height3, width3, embed_dims[2]])

    # stage 4
    x, height4, width4 = overlap_patch_embedding(feature3, embed_dims[3],
                                                 kernel_size=patch_sizes[3], strides=strides[3])
    for index in range(n_blocks[3]):
        attention_range = attention_scheduler[block_range[3]: block_range[4]]
        projection_range = projection_scheduler[block_range[3]: block_range[4]]
        x = seg_former_encoder_block(x, height4, width4, embed_dims[3], heads[3],
                                     sr_ratio=reduction_ratios[3],
                                     expansion_rate=expansion_rate[3],
                                     attention_drop_rate=attention_range[index],
                                     projection_drop_rate=projection_range[index],
                                     drop_rate=drop_rate)
    x = LayerNormalization()(x)
    feature4 = tf.reshape(x, shape=[-1, height4, width4, embed_dims[3]])

    feature1 = seg_former_decoder_block(feature1, decoder_embed_dim, up_size=(4, 4))
    feature2 = seg_former_decoder_block(feature2, decoder_embed_dim, up_size=(8, 8))
    feature3 = seg_former_decoder_block(feature3, decoder_embed_dim, up_size=(16, 16))
    feature4 = seg_former_decoder_block(feature4, decoder_embed_dim, up_size=(32, 32))

    x = seg_former_head([feature1, feature2, feature3, feature4], decoder_embed_dim, num_classes, drop_rate)
    model = Model(inputs, x, name='seg_former')

    return model


def SegFormerB0(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[2, 2, 2, 2], embed_dims=[32, 64, 120, 256],
                     decoder_embed_dim=256, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB1(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[2, 2, 2, 2], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=256, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB2(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[3, 3, 6, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB3(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[3, 3, 18, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB4(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[3, 8, 27, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[8, 8, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)


def SegFormerB5(input_shape, num_classes, attention_drop_rate=0., drop_rate=0.):
    return SegFormer(input_shape, num_classes, n_blocks=[3, 6, 40, 3], embed_dims=[64, 128, 320, 512],
                     decoder_embed_dim=768, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], heads=[1, 2, 4, 8],
                     reduction_ratios=[8, 4, 2, 1], expansion_rate=[4, 4, 4, 4],
                     attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)