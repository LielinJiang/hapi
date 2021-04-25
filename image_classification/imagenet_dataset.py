# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import math
import random
import numpy as np
from PIL import Image

import paddle
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms

class NvJpeg:
    def __init__(self):
        pass

    def __call__(self, path):
        try:
            img_bytes = paddle.io.read_file(path)
            return paddle.cast(paddle.io.decode_jpeg(img_bytes), 'float32')#.transpose([1,2,0])
        except:
            img = np.array(Image.open(path).convert('RGB')).astype('float32').transpose([1, 2, 0])
            return paddle.to_tensor(img)

class ImageNetDataset(DatasetFolder):
    def __init__(self,
                 path,
                 mode='train',
                 image_size=224,
                 resize_short_size=256,
                 use_gpu=False):
        super(ImageNetDataset, self).__init__(path)
        self.mode = mode
        self.use_gpu = use_gpu

        if self.use_gpu:
            normalize = transforms.Normalize(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375], data_format='CHW')
                # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data_format='CHW')
                # mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
            if self.mode == 'train':
                self.transform = transforms.Compose([
                    NvJpeg(),
                    transforms.RandomResizedCrop(image_size, data_format='CHW'),
                    transforms.RandomHorizontalFlip(data_format='CHW'), 
                    # transforms.ToTensor(),
                    # transforms.RandomHorizontalFlip(), transforms.ToTensor(place=None),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    NvJpeg(),
                    transforms.Resize(resize_short_size, data_format='CHW'),
                    transforms.CenterCrop(image_size, data_format='CHW'), 
                    # transforms.ToTensor(),
                    # transforms.CenterCrop(image_size), transforms.ToTensor(place=paddle.CUDAPlace(0)),
                    normalize
                ])
        else:
            normalize = transforms.Normalize(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
            if self.mode == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(), transforms.Transpose(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(resize_short_size),
                    transforms.CenterCrop(image_size), transforms.Transpose(),
                    normalize
                ])

    def __getitem__(self, idx):
        if self.use_gpu:
            img_path, label = self.samples[idx]
            return self.transform(img_path), paddle.to_tensor(label)
        else:
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            label = np.array([label]).astype(np.int64)
            return self.transform(img), label

    def __len__(self):
        return len(self.samples)
