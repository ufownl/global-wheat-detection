import os
import time
import random
import argparse
import mxnet as mx
from dataset import load_dataset, get_batches
from model import init_model, load_model


def train(max_epochs, learning_rate, batch_size, img_w, img_h, sgd, context):
    print("Loading dataset...", flush=True)
    dataset = load_dataset("data")
    split = int(len(dataset) * 0.9)
    training_set = dataset[:split]
    print("Training set: ", len(training_set))
    validation_set = dataset[split:]
    print("Validation set: ", len(validation_set))

    if os.path.isfile("model/global-wheat-yolo3-darknet53.params"):
        model = load_model("model/global-wheat-yolo3-darknet53.params", ctx=context)
    else:
        model = init_model(ctx=context)

    print("Learning rate: ", learning_rate)
    if sgd:
        print("Optimizer: SGD")
        trainer = mx.gluon.Trainer(model.collect_params(), "SGD", {
            "learning_rate": learning_rate,
            "momentum": 0.5
        })
    else:
        print("Optimizer: Nadam")
        trainer = mx.gluon.Trainer(model.collect_params(), "Nadam", {
            "learning_rate": learning_rate
        })
    if os.path.isfile("model/global-wheat-yolo3-darknet53.state"):
        trainer.load_states("model/global-wheat-yolo3-darknet53.state")

    print("Traning...", flush=True)
    for epoch in range(max_epochs):
        ts = time.time()

        random.shuffle(training_set)
        training_total_L = 0.0
        training_batches = 0
        for x, objectness, center_targets, scale_targets, weights, class_targets, gt_bboxes in get_batches(training_set, batch_size, width=img_w, height=img_h, net=model, ctx=context):
            training_batches += 1
            with mx.autograd.record():
                obj_loss, center_loss, scale_loss, cls_loss = model(x, gt_bboxes, objectness, center_targets, scale_targets, weights, class_targets)
                L = obj_loss + center_loss + scale_loss + cls_loss
                L.backward()
            trainer.step(x.shape[0])
            training_batch_L = mx.nd.mean(L).asscalar()
            if training_batch_L != training_batch_L:
                raise ValueError()
            training_total_L += training_batch_L
            print("[Epoch %d  Batch %d]  batch_loss %.10f  average_loss %.10f  elapsed %.2fs" % (
                epoch, training_batches, training_batch_L, training_total_L / training_batches, time.time() - ts
            ), flush=True)
        training_avg_L = training_total_L / training_batches

        print("[Epoch %d]  training_loss %.10f  duration %.2fs" % (
            epoch + 1, training_avg_L, time.time() - ts
        ), flush=True)

        model.save_parameters("model/global-wheat-yolo3-darknet53.params")
        trainer.save_states("model/global-wheat-yolo3-darknet53.state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a global-wheat-detection trainer.")
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 0.001)", type=float, default=0.001)
    parser.add_argument("--batch_size", help="set the batch size (default: 32)", type=int, default=32)
    parser.add_argument("--img_w", help="set the width of training images (default: 256)", type=int, default=256)
    parser.add_argument("--img_h", help="set the height of training images (default: 256)", type=int, default=256)
    parser.add_argument("--sgd", help="using sgd optimizer", action="store_true")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    train(args.max_epochs, args.learning_rate, args.batch_size, args.img_w, args.img_h, args.sgd, context)