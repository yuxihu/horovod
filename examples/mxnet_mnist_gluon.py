# Step 0: import required packages
import argparse
import logging
import time

import horovod.mxnet as hvd
import mxnet as mx
from mxnet import autograd, gluon, nd


# Training settings
parser = argparse.ArgumentParser(description='MXNet MNIST Example')
parser.add_argument('--data-dir', type=str, default='/home/ubuntu/mnist/data',
                    help='data dir to load mnist data')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='training data type (default: float32)')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs (default: 5)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(args)


# Function to get MNIST iterator
def get_mnist_iterator():
    input_shape = (1, 28, 28)
    data_dir = args.data_dir
    batch_size = args.batch_size

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx,
                                          batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx,
                                           batch_axis=0)
        return data, label

    train_iter = mx.io.MNISTIter(
        image="%s/train-images-idx3-ubyte" % data_dir,
        label="%s/train-labels-idx1-ubyte" % data_dir,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=False,
        num_parts=hvd.size(),
        part_index=hvd.rank()
    )

    val_iter = mx.io.MNISTIter(
        image="%s/t10k-images-idx3-ubyte" % data_dir,
        label="%s/t10k-labels-idx1-ubyte" % data_dir,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=False,
        flat=False
    )

    return train_iter, val_iter, batch_fn


# Function to evaluate accuracy for a model
def evaluate_accuracy(net, data_iter, context):
    data_iter.reset()

    acc = mx.metric.Accuracy()
    for _, batch in enumerate(data_iter):
        data, label = batch_fn(batch, [context])
        outputs = [net(x.astype(args.dtype, copy=False)) for x in data]
        preds = [nd.argmax(output, axis=1) for output in outputs]
        acc.update(preds, label)
    return acc.get()[1]


# Step 1: initialize Horovod
hvd.init()

# Horovod: pin GPU to local rank
context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())

# Step 2: load data
train_iter, val_iter, batch_fn = get_mnist_iterator()

# Step 3: define network
net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(512, activation="relu"))
    net.add(gluon.nn.Dense(10))
net.cast(args.dtype)
net.hybridize()

# Step 4: initialize parameters
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
net.initialize(initializer, ctx=context)

# Horovod: fetch and broadcast parameters
params = net.collect_params()
if params is not None:
    hvd.broadcast_parameters(params, root_rank=0)

# Step 5: create optimizer
optimizer_params = {'momentum': args.momentum,
                    'learning_rate': args.lr * hvd.size(),
                    'rescale_grad': 1.0 / args.batch_size}
if args.dtype == 'float16':
    optimizer_params['multi_precision'] = True
opt = mx.optimizer.create('sgd', **optimizer_params)

# Horovod: wrap optimizer with DistributedOptimizer
opt = hvd.DistributedOptimizer(opt)

# Step 6: create trainer and loss function
trainer = gluon.Trainer(params, opt, kvstore=None)
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

# Step 7: train model
for epoch in range(args.epochs):
    tic = time.time()

    train_iter.reset()
    for nbatch, batch in enumerate(train_iter, start=1):
        data, label = batch_fn(batch, [context])
        with autograd.record(): # Start recording the derivatives
            outputs = [net(x.astype(args.dtype, copy=False)) for x in data] # the forward iteration
            loss = [loss_fn(yhat, y) for yhat, y in zip(outputs, label)] # compute the loss
        for l in loss:
            l.backward() # backpropgation
        trainer.step(args.batch_size)

    train_acc = evaluate_accuracy(net, train_iter, context)
    if hvd.rank() == 0:
        elapsed = time.time() - tic
        speed = nbatch * args.batch_size * hvd.size() / elapsed
        logging.info('Epoch[%d]\tSpeed=%.2f samples/s\tTime cost=%f', epoch, speed, elapsed)

    # Evaluate model accuracy
    val_acc = evaluate_accuracy(net, val_iter, context)
    logging.info('Epoch[%d]\tTrain-accuracy=%f\tValidation-accuracy=%f', epoch, train_acc, val_acc)

    if hvd.rank() == 0 and epoch == args.epochs - 1:
        assert val_acc > 0.96, "Achieved accuracy (%f) is lower than expected (0.96)" % val_acc
