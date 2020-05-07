import os
import json
import warnings
import numpy as np
import mxnet as mx
import pandas as pd
import gluoncv as gcv
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool


def load_dataset(root):
    csv = pd.read_csv(os.path.join(root, "train.csv"))
    data = {}
    for i in csv.index:
        key = csv["image_id"][i]
        bbox = json.loads(csv["bbox"][i])
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], 0.0]
        if key in data:
            data[key].append(bbox)
        else:
            data[key] = [bbox]
    return [(os.path.join(root, "train", k + ".jpg"), v) for k, v in data.items()]

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def get_batches(dataset, batch_size, width=1024, height=1024, net=None, ctx=mx.cpu()):
    batches = len(dataset) // batch_size
    if batches * batch_size < len(dataset):
        batches += 1
    sampler = Sampler(width, height, net, ctx)
    with Pool(cpu_count() * 2) as p:
        for i in range(batches):
            start = i * batch_size
            samples = p.map(sampler, dataset[start:start+batch_size])
            stack_fn = [gcv.data.batchify.Stack()]
            pad_fn = [gcv.data.batchify.Pad(pad_val=-1)]
            if net is None:
                yield gcv.data.batchify.Tuple(*(stack_fn + pad_fn))(samples)
            else:
                yield gcv.data.batchify.Tuple(*(stack_fn * 6 + pad_fn))(samples)

def reconstruct_color(img):
    mean = mx.nd.array([0.485, 0.456, 0.406])
    std = mx.nd.array([0.229, 0.224, 0.225])
    return ((img * std + mean) * 255).astype("uint8")


class Sampler:
    def __init__(self, width, height, net=None, ctx=mx.cpu(), **kwargs):
        self._net = net
        self._ctx = ctx
        if net is None:
            self._transform = gcv.data.transforms.presets.yolo.YOLO3DefaultValTransform(width, height, **kwargs)
        else:
            self._transform = gcv.data.transforms.presets.yolo.YOLO3DefaultTrainTransform(width, height, net=net, **kwargs)

    def __call__(self, data):
        raw = load_image(data[0])
        res = self._transform(raw, np.array(data[1]))
        return [mx.nd.array(x, ctx=self._ctx) for x in res]


if __name__ == "__main__":
    data = load_dataset("./data")
    print("dataset preview: ", data[:3])
    net = gcv.model_zoo.yolo3_darknet53_custom(["wheat"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.initialize(mx.init.Xavier())
    print("training batch preview: ", next(get_batches(data, 4, net=net)))
    print("validation batch preview: ", next(get_batches(data, 4)))
    import matplotlib.pyplot as plt
    print("data visual preview: ")
    sampler = Sampler(1024, 1024, net)
    for i, x in enumerate(data):
        print(data[i][0])
        y = sampler(x)
        gcv.utils.viz.plot_bbox(reconstruct_color(y[0].transpose((1, 2, 0))), y[6])
        plt.show()
