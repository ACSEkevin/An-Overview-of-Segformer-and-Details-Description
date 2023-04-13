from keras.layers import *
import tensorflow as tf


def overlap_patch_embedding(inputs, n_filters, kernel_size, strides):
    x = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    print('overlap patch embedding, x.shape: {}'.format(x.shape))
    batches, height, width, embed_dim = x.shape
    x = tf.reshape(x, shape=[-1, height * width, embed_dim])

    return LayerNormalization()(x), height, width


def efficient_multi_head_attention(inputs, height, width, embed_dim, n_heads, scaler=None,
                                   use_bias=True, sr_ratio: int = 1,
                                   attention_drop_rate=0., projection_drop_rate=0.):
    batches, n_patches, channels = inputs.shape
    assert channels == embed_dim
    assert height and width and height * width == n_patches

    head_dim = embed_dim // n_heads
    scaler = head_dim ** -0.5 if scaler is None else scaler

    query = Dense(embed_dim, use_bias=use_bias)(inputs)
    query = tf.reshape(query, shape=[-1, n_patches, n_heads, head_dim])
    query = tf.transpose(query, perm=[0, 2, 1, 3])

    if sr_ratio > 1:
        inputs = tf.reshape(inputs, shape=[-1, height, width, embed_dim])
        # shape -> [batches, height/sr, width/sr, embed_dim]
        inputs = Conv2D(embed_dim, kernel_size=sr_ratio, strides=sr_ratio, padding='same')(inputs)
        inputs = LayerNormalization()(inputs)
        # shape -> [batches, height * width/sr ** 2, embed_dim]
        inputs = tf.reshape(inputs, shape=[-1, (height * width) // (sr_ratio ** 2), embed_dim])

    key_value = Dense(embed_dim * 2, use_bias=use_bias)(inputs)
    if sr_ratio > 1:
        key_value = tf.reshape(key_value, shape=[-1, (height * width) // (sr_ratio ** 2), 2, n_heads, head_dim])
    else:
        key_value = tf.reshape(key_value, shape=[-1, n_patches, 2, n_heads, head_dim])
    key_value = tf.transpose(key_value, perm=[2, 0, 3, 1, 4])
    key, value = key_value[0], key_value[1]

    alpha = tf.matmul(a=query, b=key, transpose_b=True) * scaler
    alpha_prime = tf.nn.softmax(alpha, axis=-1)
    alpha_prime = Dropout(rate=attention_drop_rate)(alpha_prime)

    b = tf.matmul(alpha_prime, value)
    b = tf.transpose(b, perm=[0, 2, 1, 3])
    b = tf.reshape(b, shape=[-1, n_patches, embed_dim])

    x = Dense(embed_dim, use_bias=use_bias)(b)
    x = Dropout(rate=projection_drop_rate)(x)

    return x


def mixed_feedforward_network(inputs, height, width, embed_dim, expansion_rate=4, drop_rate=0., ):
    batches, n_patches, channels = inputs.shape
    assert n_patches == height * width and channels == embed_dim

    x = Dense(int(embed_dim * expansion_rate), use_bias=True)(inputs)
    x = tf.reshape(x, shape=[-1, height, width, int(embed_dim * expansion_rate)])
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = tf.reshape(x, shape=[-1, n_patches, int(embed_dim * expansion_rate)])
    x = Activation('gelu')(x)
    x = Dense(embed_dim, use_bias=True)(x)
    x = Dropout(rate=drop_rate)(x)

    return x


def seg_former_encoder_block(inputs, height, width, embed_dim, n_heads=8, sr_ratio=1, expansion_rate=4,
                             attention_drop_rate=0., projection_drop_rate=0., drop_rate=0.):
    x = LayerNormalization()(inputs)
    x = efficient_multi_head_attention(x, height, width, embed_dim, n_heads=n_heads, sr_ratio=sr_ratio,
                                       attention_drop_rate=attention_drop_rate,
                                       projection_drop_rate=projection_drop_rate)
    branch1 = Add()([inputs, x])
    x = LayerNormalization()(branch1)
    x = mixed_feedforward_network(x, height, width, embed_dim, expansion_rate, drop_rate)
    x = Add()([branch1, x])

    return x
