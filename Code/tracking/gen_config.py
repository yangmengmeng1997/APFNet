import os
import json
import numpy as np


def gen_config(seq_path, set_type):

    path, seqname = os.path.split(seq_path)
    if 'RGBT' in set_type:
        img_list_visible = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        img_list_infrared = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
    elif 'GTOT' in set_type:
        img_list_visible = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        img_list_infrared = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        gt = np.loadtxt(seq_path + '/init.txt')
    elif 'LasHeR' in set_type:
        img_list_visible = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        img_list_infrared = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')

    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
    return img_list_visible,img_list_infrared,gt
