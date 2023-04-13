from keras.layers import *
from keras.initializers.initializers_v2 import LecunNormal, Zeros, GlorotUniform
import tensorflow as tf
import numpy as np


class PatchEmbedding(Layer):
    """
    image division to sequences (14x14)
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = 224
        self.patch_size = patch_size
        self.n_patches = img_size // patch_size
        self.embed_dim = embed_dim

        self.projection = Conv2D(embed_dim, kernel_size=patch_size, strides=self.patch_size, padding='same',
                                 kernel_initializer=LecunNormal(), bias_initializer=Zeros(), trainable=True)

    def call(self, inputs, *args, **kwargs):
        batches, height, width, channels = inputs.shape

        # x.shape -> [batches, n_patches, n_patches, embed_dim]
        x = self.projection(inputs)
        x = tf.reshape(x, shape=[batches, -1, self.embed_dim])

        return x


class OverlapPatchEmbedding(Layer):
    def __init__(self, img_size=224, patch_size=7, strides=4, embed_dim=768):
        super(OverlapPatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.side_length = img_size // patch_size
        self.n_patches = self.side_length ** 2
        self.embed_dim = embed_dim

        self.projection = Conv2D(embed_dim, kernel_size=patch_size, strides=strides, padding='same',
                                 kernel_initializer=LecunNormal(), bias_initializer=Zeros(), trainable=True)
        # self.layer_norm = LayerNormalization(epsilon=1e-5)

    def call(self, inputs, *args, **kwargs):
        x = self.projection(inputs)

        batches, height, width, embed_dim = x.shape

        x = tf.reshape(x, shape=[batches, -1, embed_dim])
        # x = self.layer_norm(x)
        return x, height, width


class PositionalEmbedding(Layer):
    """
    positional embedding without class token
    """

    def __init__(self, n_patches=14, embed_dim=768):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = None
        self.n_patches = n_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            shape=[1, self.n_patches ** 2, self.embed_dim],
            initializer=Zeros(),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs, *args, **kwargs):
        batches, n_patches, embed_dim = inputs.shape
        # assert n_patches == self.n_patches ** 2
        assert embed_dim == self.embed_dim
        return inputs + self.positional_embedding


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, n_heads=8, scaler=None, use_bias=True,
                 attention_drop_rate=0., transition_drop_rate=0., ):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.scaler = self.head_dim ** -0.5 if scaler is None else scaler

        self.query_key_value = Dense(embed_dim * 3, kernel_initializer=GlorotUniform(),
                                     use_bias=use_bias, bias_initializer=Zeros())
        self.transition = Dense(embed_dim, kernel_initializer=GlorotUniform(),
                                use_bias=use_bias, bias_initializer=Zeros())

        self.attention_drop = Dropout(attention_drop_rate)
        self.transition_drop = Dropout(transition_drop_rate)

    def call(self, inputs, *args, **kwargs):
        batches, patches, embed_dim = inputs.shape

        qkv = self.query_key_value(inputs)
        qkv = tf.reshape(qkv, shape=[batches, patches, 3, self.n_heads, self.head_dim])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        query, key, value = qkv[0], qkv[1], qkv[2]

        alpha = tf.matmul(a=query, b=key, transpose_b=True) * self.scaler
        alpha_prime = tf.nn.softmax(alpha, axis=-1)
        alpha_prime = self.attention_drop(alpha_prime)

        x = tf.matmul(alpha_prime, value)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[batches, patches, embed_dim])

        x = self.transition(x)
        x = self.transition_drop(x)
        return x


class EfficientMultiHeadAttention(Layer):
    def __init__(self, embed_dim, n_heads=8, scaler=None, use_bias=True, sr_ratio: int = 1,
                 attention_drop_rate=0., projection_drop_rate=0.):
        super(EfficientMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.sr_ratio = sr_ratio
        self.head_dim = embed_dim // n_heads
        self.scaler = self.head_dim ** -0.5 if scaler is None else scaler

        self.query = Dense(embed_dim, kernel_initializer=GlorotUniform(),
                           use_bias=use_bias, bias_initializer=Zeros())
        self.key_value = Dense(embed_dim * 2, kernel_initializer=GlorotUniform(),
                               use_bias=use_bias, bias_initializer=Zeros())
        self.projection = Dense(embed_dim, kernel_initializer=GlorotUniform(),
                                use_bias=use_bias, bias_initializer=Zeros())

        if sr_ratio > 1:
            self.sample_reduction = Conv2D(embed_dim, kernel_size=sr_ratio, strides=sr_ratio, padding='same')
            self.layer_norm = LayerNormalization(epsilon=1e-5)

        self.attention_drop = Dropout(rate=attention_drop_rate)
        self.projection_drop = Dropout(rate=projection_drop_rate)

    def call(self, inputs, height=None, width=None, *args, **kwargs):
        batches, n_patches, embed_dim = inputs.shape
        assert embed_dim == self.embed_dim
        assert height and width and height * width == n_patches

        query = self.query(inputs)
        query = tf.reshape(query, shape=[batches, n_patches, self.n_heads, self.head_dim])
        # shape -> [batches, self.n_heads, n_patches, self.head_dim]
        query = tf.transpose(query, perm=[0, 2, 1, 3])

        if self.sr_ratio > 1:
            inputs = tf.reshape(inputs, shape=[batches, height, width, embed_dim])
            # shape -> [batches, height/sr, width/sr, embed_dim]
            inputs = self.sample_reduction(inputs)
            inputs = self.layer_norm(inputs)
            # shape -> [batches, height * width/sr ** 2, embed_dim]
            inputs = tf.reshape(inputs, shape=[batches, -1, embed_dim])

        kv = self.key_value(inputs)
        # shape -> [batches, height * width/sr ** 2, 2, self.n_heads, self.head_dim]
        kv = tf.reshape(kv, shape=[batches, -1, 2, self.n_heads, self.head_dim])
        # shape -> [2, batches, self.n_heads, height * width/sr ** 2, self.head_dim]
        kv = tf.transpose(kv, perm=[2, 0, 3, 1, 4])
        key, value = kv[0], kv[1]

        # shape -> [batches, self.n_heads, n_patches, height * width/sr ** 2]
        alpha = tf.matmul(a=query, b=key, transpose_b=True) * self.scaler
        alpha_prime = tf.nn.softmax(alpha, axis=-1)
        alpha_prime = self.attention_drop(alpha_prime)

        # x.shape -> [batches, n_heads, n_patches, head_dim]
        x = tf.matmul(alpha_prime, value)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[batches, n_patches, embed_dim])

        x = self.projection(x)
        x = self.projection_drop(x)
        return x


class FeedForwardNetwork(Layer):
    def __init__(self, in_features, expansion_rate=4, drop_rate=0.):
        super(FeedForwardNetwork, self).__init__()
        self.in_units = int(in_features * expansion_rate)
        self.fully_connected0 = Dense(self.in_units, kernel_initializer=GlorotUniform(),
                                      bias_initializer=Zeros())
        self.non_linearity = Activation('gelu')
        self.fully_connected1 = Dense(in_features, kernel_initializer=GlorotUniform(),
                                      bias_initializer=Zeros())
        self.drop_out = Dropout(rate=drop_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.fully_connected0(inputs)
        x = self.non_linearity(x)
        x = self.drop_out(x)
        x = self.fully_connected1(x)
        x = self.drop_out(x)

        return x


class MixedFeedforwardNetwork(Layer):
    def __init__(self, embed_dim, expansion_rate=4., out_channels=None, drop_rate=0.):
        super(MixedFeedforwardNetwork, self).__init__()
        self.out_channels = embed_dim if not out_channels else out_channels
        self.fully_connected0 = Dense(int(embed_dim * expansion_rate), kernel_initializer=GlorotUniform(),
                                      bias_initializer=Zeros())
        self.depth_wise = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')
        self.non_linearity = Activation('gelu')
        self.fully_connected1 = Dense(self.out_channels, kernel_initializer=GlorotUniform(),
                                      bias_initializer=Zeros())
        self.dropout = Dropout(drop_rate)

    def call(self, inputs, height=None, width=None, *args, **kwargs):
        batches, n_patches, embed_dim = inputs.shape
        assert height and width and height * width == n_patches
        x = self.fully_connected0(inputs)
        x = tf.reshape(x, shape=[batches, height, width, -1])
        x = self.depth_wise(x)
        x = tf.reshape(x, shape=[batches, n_patches, -1])
        x = self.non_linearity(x)
        x = self.dropout(x)
        x = self.fully_connected1(x)
        x = self.dropout(x)
        return x


class TransformerBlock(Layer):
    def __init__(self, embed_dim, expansion_rate=4, n_heads=8, scaler=None, use_bias=True,
                 attention_drop_rate=0., transition_drop_rate=0., drop_rate=0.):
        super(TransformerBlock, self).__init__()

        self.layer_norm0 = LayerNormalization(epsilon=1e-5)
        self.multi_head_attention = MultiHeadSelfAttention(
            embed_dim=embed_dim, n_heads=n_heads, scaler=scaler, use_bias=use_bias,
            attention_drop_rate=attention_drop_rate, transition_drop_rate=transition_drop_rate
        )

        self.layer_norm1 = LayerNormalization(epsilon=1e-5)
        self.feedforward = FeedForwardNetwork(embed_dim, expansion_rate=expansion_rate, drop_rate=drop_rate)
        self.feature_add = Add()

    def call(self, inputs, *args, **kwargs):
        x = self.layer_norm0(inputs)
        x = self.multi_head_attention(x)
        x1 = self.feature_add([inputs, x])

        x = self.layer_norm1(x1)
        x = self.feedforward(x)
        x = self.feature_add([x1, x])
        return x


class SegFormerBlock(Layer):
    def __init__(self, embed_dim, expansion_rate=4, n_heads=8, scaler=None, use_bias=True, sr_ratio=1,
                 attention_drop_rate=0., projection_drop_rate=0., drop_rate=0.):
        super(SegFormerBlock, self).__init__()

        self.layer_norm0 = LayerNormalization(epsilon=1e-5)
        self.multi_head_attention = EfficientMultiHeadAttention(
            embed_dim=embed_dim, n_heads=n_heads, scaler=scaler, use_bias=use_bias, sr_ratio=sr_ratio,
            attention_drop_rate=attention_drop_rate, projection_drop_rate=projection_drop_rate
        )

        self.layer_norm1 = LayerNormalization(epsilon=1e-5)
        self.feedforward = MixedFeedforwardNetwork(embed_dim, expansion_rate=expansion_rate, drop_rate=drop_rate)
        self.feature_add = Add()

    def call(self, inputs, height=None, width=None, *args, **kwargs):
        x = self.feature_add([inputs, self.multi_head_attention(
            self.layer_norm0(inputs), height=height, width=width
        )])
        x = self.feature_add([x, self.feedforward(
            self.layer_norm1(x), height=height, width=width
        )])

        return x


# TODO
class MixVisionTransformer(Layer):
    """
    The backbone will return a list of features in each stage sequentially
    """
    def __init__(self, img_size=224, version='B0', attention_drop_rate=0.,drop_rate=0.,):
        super(MixVisionTransformer, self).__init__()
        self.patch_size = [7, 3, 3, 3]
        self.strides = [4, 2, 2, 2]
        self.scaler_factors = [1, 4, 8, 16]
        self.reduction_ratio = [8, 4, 2, 1]
        self.n_heads = [1, 2, 5, 8]
        self.expansion_rates = [8, 8, 4, 4]
        self.version_dict = {'B0': {'channels': [32, 64, 160, 256], 'n_encoder': [2, 2, 2, 2]},
                             'B1': {'channels': [64, 128, 320, 512], 'n_encoder': [2, 2, 2, 2]},
                             'B2': {'channels': [64, 128, 320, 512], 'n_encoder': [3, 3, 6, 3]},
                             'B3': {'channels': [64, 128, 320, 512], 'n_encoder': [3, 3, 18, 3]},
                             'B4': {'channels': [64, 128, 320, 512], 'n_encoder': [3, 8, 27, 3]},
                             'B5': {'channels': [64, 128, 320, 512], 'n_encoder': [3, 6, 40, 3]}}

        self.config = self.version_dict[version]

        self.overlap_embedding_list = [OverlapPatchEmbedding(img_size=img_size // self.scaler_factors[index],
                                                             patch_size=self.patch_size[index],
                                                             strides=self.strides[index],
                                                             embed_dim=self.config['channels'][index])
                                       for index in range(4)]

        drop_scheduler = np.linspace(0, drop_rate, num=sum(self.config['n_encoder']))
        attention_drop_scheduler = np.linspace(0, attention_drop_rate, num=sum(self.config['n_encoder']))

        self.stage_module_list = list()
        for index, value in enumerate(self.config['n_encoder']):
            self.stage_module_list.append([
                SegFormerBlock(embed_dim=self.config['channels'][index],
                               expansion_rate=self.expansion_rates[index] if version != 'B5' else 4,
                               sr_ratio=self.reduction_ratio[index],
                               # FIXME <scheduler not entirely used>
                               attention_drop_rate=attention_drop_scheduler[index],
                               projection_drop_rate=drop_scheduler[index])
                for _ in range(value)
            ])

        self.layer_norm_list = [LayerNormalization(epsilon=1e-5) for _ in range(4)]

    def call(self, inputs, *args, **kwargs):
        batches = inputs.shape[0]
        features = list()
        x = inputs

        for patch_embedding, segformer_blocks, layer_norm in zip(
            self.overlap_embedding_list,
            self.stage_module_list,
            self.layer_norm_list
        ):
            x, height, width = patch_embedding(x)
            for segformer_block in segformer_blocks:
                x = segformer_block(x, height=height, width=width)
            x = layer_norm(x)
            x = tf.reshape(x, shape=[batches, height, width, -1])
            features.append(x)

        return features


