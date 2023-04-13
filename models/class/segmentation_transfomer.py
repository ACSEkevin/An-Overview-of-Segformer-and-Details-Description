from kevin_cvutils.models.transformer_utils import *
from kevin_cvutils.models.setr_utils import *
from keras.models import Model
import numpy as np


def SETR_(num_classes, n_blocks=24, n_heads=8, drop_rate=0.,
          attention_drop_rate=0.2, transition_drop_rate=0., include_top=True):
    """

    :param transition_drop_rate:
    :param attention_drop_rate:
    :param num_classes:
    :param n_blocks:
    :param n_heads:
    :param drop_rate:
    :param include_top:
    :return:
    """
    attention_drop_scheduler = np.linspace(0, attention_drop_rate, num=n_blocks)
    decoders = [TransformerBlock(embed_dim=768,
                                 expansion_rate=4,
                                 n_heads=n_heads,
                                 attention_drop_rate=attention_drop_scheduler[index],
                                 transition_drop_rate=transition_drop_rate) for index in range(n_blocks)]


class SETR(Model):
    def __init__(self, num_classes, n_blocks=24, n_heads=16, decoder='naive', patch_size=16, embed_dim=768,
                 drop_rate=0., attention_drop_rate=0.2, transition_drop_rate=0., include_top=True):
        super(SETR, self).__init__()
        self.num_classes = num_classes
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.decoder_key = decoder
        self.include_top = include_top

        self.patch_merging = PatchEmbedding(img_size=128, patch_size=patch_size, embed_dim=embed_dim)
        self.n_patches = self.patch_merging.n_patches
        self.positional_embedding = PositionalEmbedding(n_patches=self.n_patches, embed_dim=embed_dim)

        self.attention_drop_scheduler = np.linspace(0, attention_drop_rate, num=n_blocks)
        self.encoder_blocks = [TransformerBlock(embed_dim=embed_dim,
                                                expansion_rate=4,
                                                n_heads=n_heads,
                                                attention_drop_rate=self.attention_drop_scheduler[index],
                                                transition_drop_rate=transition_drop_rate)
                               for index in range(n_blocks)]

        self.decoder_choice = {'naive': NaiveDecoder(embed_dim, num_classes, interpolation='bilinear'),
                               'pup': ProgressiveUpSampling(embed_dim, num_classes,interpolation='bilinear'),
                               'mla': MultiLayerAggregation(embed_dim, num_classes, interpolation='bilinear')}
        self.mla_features = list()
        self.decoder = self.decoder_choice[decoder]

        self.reshape_volume = Reshape(target_shape=[self.n_patches, self.n_patches, self.embed_dim])
        self.dropout = Dropout(rate=drop_rate)

    def call(self, inputs, training=None, mask=None):
        x = self.patch_merging(inputs)
        x = self.positional_embedding(x)
        x = self.dropout(x)

        for index, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            if self.decoder_key == 'mla':
                if (index + 1) % (self.n_blocks // 4) == 0:
                    self.mla_features.append(x)
                    print('feature append {} ith'.format(index))

        if self.decoder_key in ['naive', 'pup']:
            x = self.reshape_volume(x)
            x = self.decoder(x)
        elif self.decoder_key == 'mla':
            self.mla_features.reverse()
            print('length of mla features: {}'.format(len(self.mla_features)))

            for index in range(len(self.mla_features[: 4])):
                self.mla_features[index] = self.reshape_volume(self.mla_features[index])
                print(self.mla_features[index])
            x = self.decoder(self.mla_features[: 4])

        return x


