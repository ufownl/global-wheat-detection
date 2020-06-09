import random
import argparse
import numpy as np
import mxnet as mx
import gluoncv as gcv
import matplotlib.pyplot as plt
from model import load_model
from dataset import load_image
from ensemble_boxes import *


def test(images, threshold, img_s, context):
    print("Loading model...")
    model = load_model("model/global-wheat-yolo3-darknet53.params", ctx=context)
    for path in images:
        print(path)
        raw = load_image(path)
        rh, rw, _ = raw.shape
        classes_list = []
        scores_list = []
        bboxes_list = []
        for _ in range(5):
            img, flips = gcv.data.transforms.image.random_flip(raw, px=0.5, py=0.5)
            x, _ = gcv.data.transforms.presets.yolo.transform_test(img, short=img_s)
            _, _, xh, xw = x.shape
            rot = random.randint(0, 3)
            if rot > 0:
                x = np.rot90(x.asnumpy(), k=rot, axes=(2, 3))
            classes, scores, bboxes = model(mx.nd.array(x, ctx=context))
            if rot > 0:
                if rot == 1:
                    raw_bboxes = bboxes.copy()
                    bboxes[0, :, [0, 2]] = xh - raw_bboxes[0, :, [1, 3]]
                    bboxes[0, :, [1, 3]] = raw_bboxes[0, :, [2, 0]]
                elif rot == 2:
                    bboxes[0, :, [0, 1, 2, 3]] = mx.nd.array([[xw], [xh], [xw], [xh]], ctx=context) - bboxes[0, :, [2, 3, 0, 1]]
                elif rot == 3:
                    raw_bboxes = bboxes.copy()
                    bboxes[0, :, [0, 2]] = raw_bboxes[0, :, [1, 3]]
                    bboxes[0, :, [1, 3]] = xw - raw_bboxes[0, :, [2, 0]]
                raw_bboxes = bboxes.copy()
                bboxes[0, :, 0] = raw_bboxes[0, :, [0, 2]].min(axis=0)
                bboxes[0, :, 1] = raw_bboxes[0, :, [1, 3]].min(axis=0)
                bboxes[0, :, 2] = raw_bboxes[0, :, [0, 2]].max(axis=0)
                bboxes[0, :, 3] = raw_bboxes[0, :, [1, 3]].max(axis=0)
            bboxes[0, :, :] = gcv.data.transforms.bbox.flip(bboxes[0, :, :], (xw, xh), flip_x=flips[0], flip_y=flips[1])
            bboxes[0, :, 0::2] = (bboxes[0, :, 0::2] / (xw - 1)).clip(0.0, 1.0)
            bboxes[0, :, 1::2] = (bboxes[0, :, 1::2] / (xh - 1)).clip(0.0, 1.0)
            classes_list.append([
                int(classes[0, i].asscalar()) for i in range(classes.shape[1])
                    if classes[0, i].asscalar() >= 0.0

            ])
            scores_list.append([
                scores[0, i].asscalar() for i in range(classes.shape[1])
                    if classes[0, i].asscalar() >= 0.0

            ])
            bboxes_list.append([
                bboxes[0, i].asnumpy().tolist() for i in range(classes.shape[1])
                    if classes[0, i].asscalar() >= 0.0
            ])
        bboxes, scores, classes = weighted_boxes_fusion(bboxes_list, scores_list, classes_list)
        bboxes[:, 0::2] *= rw - 1
        bboxes[:, 1::2] *= rh - 1
        gcv.utils.viz.plot_bbox(raw, [
            bboxes[i] for i in range(classes.shape[0])
                if model.classes[int(classes[i])] == "wheat" and scores[i] > threshold
        ])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a global-wheat-detection tester.")
    parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
    parser.add_argument("--threshold", help="set the positive threshold (default: 0.5)", type=float, default=0.5)
    parser.add_argument("--img_s", help="set the size of image short side", type=int, default=512)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    test(args.images, args.threshold, args.img_s, context)
