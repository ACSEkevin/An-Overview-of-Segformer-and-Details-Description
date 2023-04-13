from keras.models import Model
from kevin_cvutils.models.transformer_utils import *
from kevin_cvutils.models.segformer_utils import *


class SegFormer(Model):
    def __init__(self, num_classes, img_size=224, version='B0', attention_drop_rate=0., drop_rate=0.):
        super(SegFormer, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.scalers = [4, 8, 16, 32]
        self.encoder = MixVisionTransformer(img_size=img_size, version=version,
                                            attention_drop_rate=attention_drop_rate, drop_rate=drop_rate)

        self.decoder = SegFormerDecoder(num_classes, version=version, drop_rate=drop_rate)

    def call(self, inputs, training=None, mask=None):
        features = self.encoder(inputs)
        x = self.decoder(features)

        assert len(features) == 4
        for index in range(len(features)):
            assert features[index].shape[1] == self.img_size // self.scalers[index]
            assert features[index].shape[2] == self.img_size // self.scalers[index]

        return x
