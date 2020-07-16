#  Copyright 2020 RangerUFO
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import warnings
import mxnet as mx
import gluoncv as gcv


def init_model(ctx=mx.cpu()):
    net = gcv.model_zoo.yolo3_darknet53_custom(["wheat"], transfer="voc", ctx=ctx)
    net.set_nms(post_nms=150)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.initialize(mx.init.Xavier(), ctx=ctx)
    return net

def load_model(path, ctx=mx.cpu()):
    net = gcv.model_zoo.yolo3_darknet53_custom(["wheat"], pretrained_base=False)
    net.set_nms(post_nms=150)
    net.load_parameters(path, ctx=ctx)
    return net
