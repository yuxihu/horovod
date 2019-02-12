# Step 0: import required packages
import argparse
import logging

import horovod.mxnet as hvd
import mxnet as mx


# Training settings
parser = argparse.ArgumentParser(description='MXNet MNIST Example')
parser.add_argument('--data-dir', type=str, default='/home/ubuntu/mnist/data',
                    help='data dir to load mnist data')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='training data type (default: float32)')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs (default: 10)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(args)


# Function to get mnist iterator given a rank
def get_mnist_iterator():
    input_shape = (1, 28, 28)
    data_dir = args.data_dir
    batch_size = args.batch_size

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
        flat=False
    )

    return train_iter, val_iter

# Step 1: initialize Horovod
hvd.init()

# Horovod: pin GPU to local rank
context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())

# Step 2: load data
train_iter, val_iter = get_mnist_iterator()

# Step 3: define network
def conv_net():
    # placeholder for data
    data = mx.sym.var('data')
    # first conv layer
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=10)
    relu1 = mx.sym.Activation(data=conv1, act_type='relu')
    pool1 = mx.sym.Pooling(data=relu1, pool_type='max', kernel=(2, 2),
                           stride=(2, 2))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=20)
    relu2 = mx.sym.Activation(data=conv2, act_type='relu')
    pool2 = mx.sym.Pooling(data=relu2, pool_type='max', kernel=(2, 2),
                           stride=(2, 2))
    # first fully connected layer
    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=50)
    relu3 = mx.sym.Activation(data=fc1, act_type='relu')
    # second fully connected layer
    fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=10)
    # softmax loss
    loss = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    return loss

net = conv_net()
model = mx.mod.Module(symbol=net, context=context)
model.bind(data_shapes=train_iter.provide_data,
           label_shapes=train_iter.provide_label)

# Step 4: initialize parameters
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                             magnitude=2)
model.init_params(initializer)

# Horovod: fetch and broadcast parameters
(arg_params, aux_params) = model.get_params()
if arg_params is not None:
    hvd.broadcast_parameters(arg_params, root_rank=0)
if aux_params is not None:
    hvd.broadcast_parameters(aux_params, root_rank=0)
model.set_params(arg_params=arg_params, aux_params=aux_params)

# Step 5: create optimizer
optimizer_params = {'momentum': args.momentum,
                    'learning_rate': args.lr * hvd.size(),
                    'rescale_grad': 1.0 / args.batch_size}
if args.dtype == 'float16':
    optimizer_params['multi_precision'] = True
opt = mx.optimizer.create('sgd', **optimizer_params)

# Horovod: wrap optimizer with DistributedOptimizer
opt = hvd.DistributedOptimizer(opt)

# Step 6: train model
batch_callback = None
if hvd.rank() == 0:
    batch_callback = mx.callback.Speedometer(args.batch_size * hvd.size())
model.fit(train_iter,  # train data
          kvstore=None,  # no kvstore
          eval_data=val_iter,  # validation data
          optimizer=opt,  # use SGD to train
          eval_metric='acc',  # report accuracy during training
          batch_end_callback=batch_callback,
          num_epoch=args.epochs)  # train for at most 10 dataset passes

# Step 7: evaluate model accuracy
acc = mx.metric.Accuracy()
model.score(val_iter, acc)

if hvd.rank() == 0:
    assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than \
                                expected (0.96)" % acc.get()[1]
