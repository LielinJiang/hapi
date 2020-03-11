from __future__ import division
from __future__ import print_function

import math
import paddle.fluid as fluid

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.container import Sequential

from model import Model

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


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
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
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        x = self.conv0(inputs)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        x = fluid.layers.elementwise_add(x=short, y=conv2)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(x)
        # return fluid.layers.relu(x)


class ResNet(Model):
    def __init__(self, depth=50, num_classes=1000):
        super(ResNet, self).__init__()

        layer_config = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        assert depth in layer_config.keys(), \
            "supported depth are {} but input layer is {}".format(
                layer_config.keys(), depth)

        layers = layer_config[depth]
        num_in = [64, 256, 512, 1024]
        num_out = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        self.layers = []
        for idx, num_blocks in enumerate(layers):
            blocks = []
            shortcut = False
            for b in range(num_blocks):
                block = BottleneckBlock(
                    num_channels=num_in[idx] if b == 0 else num_out[idx] * 4,
                    num_filters=num_out[idx],
                    stride=2 if b == 0 and idx != 0 else 1,
                    shortcut=shortcut)
                blocks.append(block)
                shortcut = True
            layer = self.add_sublayer(
                "layer_{}".format(idx),
                Sequential(*blocks))
            self.layers.append(layer)

        self.global_pool = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self.fc_input_dim = num_out[-1] * 4 * 1 * 1
        self.fc = Linear(self.fc_input_dim,
                         num_classes,
                         act='softmax',
                         param_attr=fluid.param_attr.ParamAttr(
                             initializer=fluid.initializer.Uniform(
                                 -stdv, stdv)))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = fluid.layers.reshape(x, shape=[-1, self.fc_input_dim])
        x = self.fc(x)
        return x