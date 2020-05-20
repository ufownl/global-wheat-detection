import os
import cv2
import copy
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


class YOLO3TrainTransform:
    def __init__(self, width, height, net, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), **kwargs):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

        # in case network has reset_ctx to gpu
        self._fake_x = mx.nd.zeros((1, 3, height, width))
        net = copy.deepcopy(net)
        net.collect_params().reset_ctx(None)
        with mx.autograd.train_mode():
            _, self._anchors, self._offsets, self._feat_maps, _, _, _, _ = net(self._fake_x)
        self._target_generator = gcv.model_zoo.yolo.yolo_target.YOLOV3PrefetchTargetGenerator(
            num_class=len(net.classes), **kwargs)

    def __call__(self, img, label):
        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = gcv.data.transforms.image.random_expand(img, max_ratio=1.5, fill=114, keep_ratio=False)
            bbox = gcv.data.transforms.bbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = gcv.data.transforms.experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = gcv.data.transforms.image.imresize(img, self._width, self._height, interp=interp)
        bbox = gcv.data.transforms.bbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal&vertical flip
        h, w, _ = img.shape
        img, flips = gcv.data.transforms.image.random_flip(img, px=0.5, py=0.5)
        bbox = gcv.data.transforms.bbox.flip(bbox, (w, h), flip_x=flips[0], flip_y=flips[1])

        # random color jittering
        img = gcv.data.transforms.experimental.image.random_color_distort(img)

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        objectness, center_targets, scale_targets, weights, class_targets = self._target_generator(
            self._fake_x, self._feat_maps, self._anchors, self._offsets,
            gt_bboxes, gt_ids, None)
        return (img, objectness[0], center_targets[0], scale_targets[0], weights[0],
                class_targets[0], gt_bboxes[0])


class Sampler:
    def __init__(self, dataset, width, height, net=None, **kwargs):
        self._dataset = dataset
        if net is None:
            self._training_mode = False
            self._transform = gcv.data.transforms.presets.yolo.YOLO3DefaultValTransform(width, height, **kwargs)
        else:
            self._training_mode = True
            self._transform = YOLO3TrainTransform(width, height, net, **kwargs)

    def __call__(self, idx):
        if self._training_mode:
            raw, bboxes = self._load_mosaic(idx)
            blur = random.randint(0, 3)
            if blur > 0:
                raw = gauss_blur(raw, blur)
            raw = gauss_noise(raw)
            raw = mx.nd.array(raw)
        else:
            raw = load_image(self._dataset[idx][1])
            bboxes = np.array(self._dataset[idx][2])
        res = self._transform(raw, bboxes)
        return [mx.nd.array(x) for x in res]

    def _load_mosaic(self, idx):
        while True:
            xc, yc = [random.uniform(0.5, 1.5) for _ in range(2)]
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]
            for i, idx in enumerate(indices):
                img = load_image(self._dataset[idx][1]).asnumpy()
                h, w, c = img.shape
                bbs = np.array(self._dataset[idx][2])
                if i == 0:
                    mosaic = np.full((h * 2, w * 2, c), 114, np.uint8)
                    mosaic_bbs = []
                    xa1, ya1, xa2, ya2 = max(int(xc * w) - w, 0), max(int(yc * h) - h, 0), int(xc * w), int(yc * w)
                    xb1, yb1, xb2, yb2 = w - (xa2 - xa1), h - (ya2 - ya1), w, h
                elif i == 1:
                    xa1, ya1, xa2, ya2 = int(xc * w), max(int(yc * h) - h, 0), min(int(xc * w) + w, w * 2), int(yc * h)
                    xb1, yb1, xb2, yb2 = 0, h - (ya2 - ya1), min(w, xa2 - xa1), h
                elif i == 2:
                    xa1, ya1, xa2, ya2 = max(int(xc * w) - w, 0), int(yc * h), int(xc * w), min(int(yc * h) + h, h * 2)
                    xb1, yb1, xb2, yb2 = w - (xa2 - xa1), 0, max(int(xc * w), w), min(ya2 - ya1, h)
                elif i == 3:
                    xa1, ya1, xa2, ya2 = int(xc * w), int(yc * h), min(int(xc * w) + w, w * 2), min(int(yc * h) + h, h * 2)
                    xb1, yb1, xb2, yb2 = 0, 0, min(w, xa2 - xa1), min(h, ya2 - ya1)
                mosaic[ya1:ya2, xa1:xa2] = img[yb1:yb2, xb1:xb2]
                bbs[:, 0:4:2] = np.clip(bbs[:, 0:4:2] + xa1 - xb1, 0, w * 2)
                bbs[:, 1:4:2] = np.clip(bbs[:, 1:4:2] + ya1 - yb1, 0, h * 2)
                mosaic_bbs.append(bbs)
            if len(mosaic_bbs) > 0:
                return mosaic, np.concatenate(mosaic_bbs)


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
