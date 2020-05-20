import os
import cv2
import json
import random
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
    return sorted(
        [(k, os.path.join(root, "train", k + ".jpg"), v) for k, v in data.items()],
        key=lambda x: x[0]
    )

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def get_batches(dataset, batch_size, width=512, height=512, net=None, ctx=mx.cpu()):
    batches = len(dataset) // batch_size
    sampler = Sampler(dataset, width, height, net)
    with Pool(cpu_count() * 2) as p:
        for i in range(batches):
            start = i * batch_size
            samples = p.map(sampler, range(start, start + batch_size))
            stack_fn = [gcv.data.batchify.Stack()]
            pad_fn = [gcv.data.batchify.Pad(pad_val=-1)]
            if net is None:
                batch = gcv.data.batchify.Tuple(*(stack_fn + pad_fn))(samples)
            else:
                batch = gcv.data.batchify.Tuple(*(stack_fn * 6 + pad_fn))(samples)
            yield [x.as_in_context(ctx) for x in batch]

def gauss_blur(image, level):
    return cv2.blur(image, (level * 2 + 1, level * 2 + 1))

def gauss_noise(image):
    for i in range(image.shape[2]):
        c = image[:, :, i]
        diff = 255 - c.max();
        noise = np.random.normal(0, random.randint(1, 6), c.shape)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = diff * noise
        image[:, :, i] = c + noise.astype(np.uint8)
    return image


class Sampler:
    def __init__(self, dataset, width, height, net=None, **kwargs):
        self._dataset = dataset
        if net is None:
            self._training_mode = False
            self._transform = gcv.data.transforms.presets.yolo.YOLO3DefaultValTransform(width, height, **kwargs)
        else:
            self._training_mode = True
            self._transform = gcv.data.transforms.presets.yolo.YOLO3DefaultTrainTransform(width, height, net=net, **kwargs)

    def __call__(self, idx):
        raw = load_image(self._dataset[idx][1])
        bboxes = np.array(self._dataset[idx][2])
        if self._training_mode:
            raw = raw.asnumpy()
            blur = random.randint(0, 3)
            if blur > 0:
                raw = gauss_blur(raw, blur)
            raw = gauss_noise(raw)
            raw = mx.nd.array(raw)
            h, w, _ = raw.shape
            raw, flips = gcv.data.transforms.image.random_flip(raw, py=0.5)
            bboxes = gcv.data.transforms.bbox.flip(bboxes, (w, h), flip_y=flips[1])
        res = self._transform(raw, bboxes)
        return [mx.nd.array(x) for x in res]


def reconstruct_color(img):
    mean = mx.nd.array([0.485, 0.456, 0.406])
    std = mx.nd.array([0.229, 0.224, 0.225])
    return ((img * std + mean).clip(0.0, 1.0) * 255).astype("uint8")


if __name__ == "__main__":
    from model import init_model
    net = init_model()
    data = load_dataset("data")
    print("dataset preview: ", data[:3])
    print("max count of bboxes: ", max([len(bboxes) for _, _, bboxes in data]))
    print("training batch preview: ", next(get_batches(data, 4, net=net)))
    print("validation batch preview: ", next(get_batches(data, 4)))
    import matplotlib.pyplot as plt
    print("data visual preview: ")
    sampler = Sampler(data, 512, 512, net)
    for i, x in enumerate(data):
        print(x[1])
        y = sampler(i)
        gcv.utils.viz.plot_bbox(reconstruct_color(y[0].transpose((1, 2, 0))), y[6])
        plt.show()
