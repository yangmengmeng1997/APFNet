import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import sys
sys.path.insert(0,'/home/zhuli/xuewanlin/cat/MDNet_CAT_SK_Trans/modules/')
from sample_generator import SampleGenerator
from utils import crop_image2


class RegionDataset(data.Dataset):
    def __init__(self, img_list_v, img_list_i, gt, opts):
        self.img_list_v = np.asarray(img_list_v)
        self.img_list_i = np.asarray(img_list_i)
        self.gt = gt

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.flip = opts.get('flip', False)
        self.rotate = opts.get('rotate', 0)
        self.blur = opts.get('blur', 0)

        self.index = np.random.permutation(len(self.img_list_v))
        self.pointer = 0

        image_v = Image.open(self.img_list_v[0]).convert('RGB')
        self.pos_generator = SampleGenerator('uniform', image_v.size,
                opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image_v.size,
                opts['trans_neg'], opts['scale_neg'])

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list_v))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list_v))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions_v = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions_v = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        pos_regions_i = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions_i = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        for i, (img_path_v, img_path_i, bbox) in enumerate(zip(self.img_list_v[idx], self.img_list_i[idx], self.gt[idx])):
            image_v = Image.open(img_path_v).convert('RGB')
            image_v = np.asarray(image_v)
            image_i = Image.open(img_path_i).convert('RGB')
            image_i = np.asarray(image_i)

            n_pos = (self.batch_pos - len(pos_regions_v)) // (self.batch_frames - i)
            n_neg = (self.batch_neg - len(neg_regions_v)) // (self.batch_frames - i)
            pos_examples = self.pos_generator(bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = self.neg_generator(bbox, n_neg, overlap_range=self.overlap_neg)

            pos_regions_v = np.concatenate((pos_regions_v, self.extract_regions(image_v, pos_examples)), axis=0)
            neg_regions_v = np.concatenate((neg_regions_v, self.extract_regions(image_v, neg_examples)), axis=0)
            
            pos_regions_i = np.concatenate((pos_regions_i, self.extract_regions(image_i, pos_examples)), axis=0)
            neg_regions_i = np.concatenate((neg_regions_i, self.extract_regions(image_i, neg_examples)), axis=0)

        pos_regions_v = torch.from_numpy(pos_regions_v)
        neg_regions_v = torch.from_numpy(neg_regions_v)
        pos_regions_i = torch.from_numpy(pos_regions_i)
        neg_regions_i = torch.from_numpy(neg_regions_i)
        return pos_regions_v, neg_regions_v, pos_regions_i, neg_regions_i

    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image2(image, sample, self.crop_size, self.padding,
                    self.flip, self.rotate, self.blur)
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions
