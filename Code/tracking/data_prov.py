import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
sys.path.insert(0, '../modules')
from utils import crop_image2
import pdb

class RegionExtractor():
    def __init__(self, image_v, image_i, samples, opts):
        self.image_v = np.asarray(image_v)
        self.image_i = np.asarray(image_i)
        self.samples = samples

        self.crop_size = opts['img_size']
        self.padding = opts['padding']
        self.batch_size = opts['batch_test']

        self.index = np.arange(len(samples))
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions_v, regions_i = self.extract_regions(index)
            #pdb.set_trace()
            regions_v = torch.from_numpy(regions_v)
            regions_i = torch.from_numpy(regions_i)
            return regions_v, regions_i
    next = __next__

    def extract_regions(self, index):
        regions_v = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        regions_i = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions_v[i] = crop_image2(self.image_v, sample, self.crop_size, self.padding)
            regions_i[i] = crop_image2(self.image_i, sample, self.crop_size, self.padding)
        regions_v = regions_v.transpose(0, 3, 1, 2)
        regions_v = regions_v.astype('float32') - 128.
        regions_i = regions_i.transpose(0, 3, 1, 2)
        regions_i = regions_i.astype('float32') - 128.
        return regions_v, regions_i
