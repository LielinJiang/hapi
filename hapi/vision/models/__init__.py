#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from . import resnet
from . import vgg
from . import mobilenetv1
from . import mobilenetv2
from . import darknet
from . import yolov3
from . import tsm
from . import bmn_model

from .resnet import *
from .mobilenetv1 import *
from .mobilenetv2 import *
from .vgg import *
from .darknet import *
from .yolov3 import *
from .tsm import *
from .bmn_model import *

__all__ = resnet.__all__ \
        + vgg.__all__ \
        + mobilenetv1.__all__ \
        + mobilenetv2.__all__ \
        + darknet.__all__ \
        + yolov3.__all__ \
        + tsm.__all__ \
        + bmn_model.__all__
