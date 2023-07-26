#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/1/12 上午10:00
# @Author      :weiz
# @ProjectName :insightface
# @File        :rec2image.py
# @Description :
import mxnet as mx
import mxnet.ndarray as nd
from skimage import io
import numpy as np
from mxnet import recordio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

path_prefix = '/home/dell/Desktop/faces_umd/train'
path_imgidx = '/home/dell/Desktop/faces_umd/train.idx'
path_imgrec = '/home/dell/Desktop/faces_umd/train.rec'
output_dir = '/home/dell/Desktop/faces_umd/adaFace'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

for i in tqdm(range(501195)):
    header, s = recordio.unpack(imgrec.read_idx(i + 1))
    img = mx.image.imdecode(s).asnumpy()
    if img is None:
        continue
    label = str(header.label)
    id = str(i)

    label_dir = os.path.join(output_dir, label)
    # 检查标签文件夹是否存在
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    # plt.imshow(img)
    # plt.title('id=' + str(i) + 'label=' + str(header.label))
    # plt.pause(0.1)
    # print('id=' + str(i) + 'label=' + str(header.label))

    fname = 'Figure_{}.jpg'.format(id)
    fpath = os.path.join(label_dir, fname)
    io.imsave(fpath, img)
