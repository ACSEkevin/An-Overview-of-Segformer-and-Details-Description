from keras.utils import Sequence, load_img, img_to_array
from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np


class SegmentationDataGenerator(Sequence):
    def __init__(self, img_dir, mask_dir, img_size=(448, 448), batch_size=32, limit=6000):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.limit = limit

        self.img_filename = os.listdir(self.img_dir)[: limit]
        self.mask_filename = os.listdir(self.mask_dir)[: limit]

    def __len__(self):
        """
        :return: number of batches
        """
        return int(np.ceil(len(self.img_filename) / self.batch_size))

    def __getitem__(self, index):
        img_name_batch = self.img_filename[index * self.batch_size: (index + 1) * self.batch_size]
        mask_name_batch = self.mask_filename[index * self.batch_size: (index + 1) * self.batch_size]
        img_batch = np.array([self.process_img(os.path.join(self.img_dir, filename), color_mode='rgb')
                              for filename in img_name_batch])
        mask_batch = np.array([self.process_img(os.path.join(self.mask_dir, filename), color_mode='grayscale')
                               for filename in mask_name_batch])
        return img_batch, mask_batch

    def process_img(self, path, color_mode='rgb'):
        pil_img = load_img(path, color_mode=color_mode, target_size=self.img_size)
        return img_to_array(pil_img) / 255

    @property
    def input_shape(self):
        return self.img_size.__add__((3, ))

    @property
    def limit_images(self):
        return self.limit


train_img_dir = 'ADEChallengeData2016/images/training'
train_mask_dir = 'ADEChallengeData2016/annotations/training'
val_img_dir = 'ADEChallengeData2016/images/validation'
val_mask_dir = 'ADEChallengeData2016/annotations/validation'

train_gen = SegmentationDataGenerator(train_img_dir, train_mask_dir, img_size=(224, 224))
val_gen = SegmentationDataGenerator(val_img_dir, val_mask_dir, img_size=(224, 224))

# val_img_list = os.listdir(val_img_dir)[: 1000]
# val_mask_list = os.listdir(val_mask_dir)[: 1000]
#
# val_imgs = np.array([resize(imread(os.path.join(val_img_dir, filename)), output_shape=(224, 224)) / 255
#                      for filename in val_img_list])
# val_masks = np.array([resize(imread(os.path.join(val_mask_dir, filename)), output_shape=(224, 224)) / 255
#                       for filename in val_mask_list])
# val_masks = np.expand_dims(val_masks, axis=-1)
# print(val_imgs.shape)
# print(val_masks.shape)
