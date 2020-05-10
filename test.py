import argparse
import mxnet as mx
import gluoncv as gcv
import matplotlib.pyplot as plt
from model import load_model
from dataset import load_image


def test(images, threshold, img_s, context):
    print("Loading model...")
    model = load_model("model/global-wheat-yolo3-darknet53.params", ctx=context)
    for path in images:
        print(path)
        raw = load_image(path)
        x, _ = gcv.data.transforms.presets.yolo.transform_test(raw, short=img_s)
        classes, scores, bboxes = model(x.as_in_context(context))
        bboxes[0, :, 0::2] = bboxes[0, :, 0::2] / x.shape[3] * raw.shape[1]
        bboxes[0, :, 1::2] = bboxes[0, :, 1::2] / x.shape[2] * raw.shape[0]
        gcv.utils.viz.plot_bbox(raw, [
            bboxes[0, i].asnumpy() for i in range(classes.shape[1])
                if model.classes[int(classes[0, i].asscalar())] == "wheat" and scores[0, i].asscalar() > threshold
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
