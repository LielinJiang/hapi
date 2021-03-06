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

from __future__ import division
from __future__ import print_function

import math
import paddle.fluid as fluid

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.container import Sequential

from hapi.model import Model
from hapi.download import get_weights_path

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

model_urls = {
    'resnet50': ('https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
                 '0884c9087266496c41c60d14a96f8530')
}


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._batch_norm(x)

        return x


class BasicBlock(fluid.dygraph.Layer):

    expansion = 1

    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BasicBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = short + conv1

        return fluid.layers.relu(y)


class BottleneckBlock(fluid.dygraph.Layer):

    expansion = 4

    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * self.expansion,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * self.expansion,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * self.expansion

    def forward(self, inputs):
        x = self.conv0(inputs)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        x = fluid.layers.elementwise_add(x=short, y=conv2)

        return fluid.layers.relu(x)


class ResNet(Model):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.
        classifier_activation (str): activation for the last fc layer. Default: 'softmax'.
    """

    def __init__(self,
                 Block,
                 depth=50,
                 num_classes=1000,
                 with_pool=True,
                 classifier_activation='softmax'):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.with_pool = with_pool

        layer_config = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        assert depth in layer_config.keys(), \
            "supported depth are {} but input layer is {}".format(
                layer_config.keys(), depth)

        layers = layer_config[depth]

        in_channels = 64
        out_channels = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.layers = []
        for idx, num_blocks in enumerate(layers):
            blocks = []
            shortcut = False
            for b in range(num_blocks):
                if b == 1:
                    in_channels = out_channels[idx] * Block.expansion
                block = Block(
                    num_channels=in_channels,
                    num_filters=out_channels[idx],
                    stride=2 if b == 0 and idx != 0 else 1,
                    shortcut=shortcut)
                blocks.append(block)
                shortcut = True
            layer = self.add_sublayer("layer_{}".format(idx),
                                      Sequential(*blocks))
            self.layers.append(layer)

        if with_pool:
            self.global_pool = Pool2D(
                pool_size=7, pool_type='avg', global_pooling=True)

        if num_classes > 0:
            stdv = 1.0 / math.sqrt(out_channels[-1] * Block.expansion * 1.0)
            self.fc_input_dim = out_channels[-1] * Block.expansion * 1 * 1
            self.fc = Linear(
                self.fc_input_dim,
                num_classes,
                act=classifier_activation,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        for layer in self.layers:
            x = layer(x)

        if self.with_pool:
            x = self.global_pool(x)

        if self.num_classes > -1:
            x = fluid.layers.reshape(x, shape=[-1, self.fc_input_dim])
            x = self.fc(x)
        return x


def _resnet(arch, Block, depth, pretrained):
    model = ResNet(Block, depth, num_classes=1000, with_pool=True)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path(model_urls[arch][0],
                                       model_urls[arch][1])
        assert weight_path.endswith(
            '.pdparams'), "suffix of weight must be .pdparams"
        model.load(weight_path[:-9])
    return model


def resnet18(pretrained=False):
    """ResNet 18-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet('resnet18', BasicBlock, 18, pretrained)


def resnet34(pretrained=False):
    """ResNet 34-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet('resnet34', BasicBlock, 34, pretrained)


def resnet50(pretrained=False):
    """ResNet 50-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet('resnet50', BottleneckBlock, 50, pretrained)


def resnet101(pretrained=False):
    """ResNet 101-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet('resnet101', BottleneckBlock, 101, pretrained)


def resnet152(pretrained=False):
    """ResNet 152-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet('resnet152', BottleneckBlock, 152, pretrained)
