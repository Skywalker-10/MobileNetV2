import mindspore_hub as mshub
import mindspore
from mindspore import context, Tensor, nn
from mindspore.train.model import Model
from mindspore.common import dtype as mstype
from mindspore.dataset.transforms import py_transforms
import os
from mindspore.nn import Momentum
from mindspore import save_checkpoint, load_checkpoint,load_param_into_net
from mindspore import ops
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C
from mindspore import Model


context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    device_id=0)

model = "mindspore/ascend/1.1/mobilenetv2_v1.1"
network = mshub.load(model, num_classes=1000,include_top=False)
network.set_train(False)


class ReduceMeanFlatten(nn.Cell):
    def __init__(self):
        super(ReduceMeanFlatten, self).__init__()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.mean(x, (2, 3))
        x = self.flatten(x)
        return x


last_channel = 1280

num_classes = 2388

reducemean_flatten = ReduceMeanFlatten()

classification_layer = nn.Dense(last_channel, num_classes)
classification_layer.set_train(True)

train_network = nn.SequentialCell([network, reducemean_flatten, classification_layer])



def create_dataset_rp2k(do_train):
    if do_train:
        data_set = ds.ImageFolderDataset("/data/RP2K/train", num_parallel_workers=32, shuffle=do_train)
    else:
        data_set = ds.ImageFolderDataset("/data/RP2K/test", num_parallel_workers=32, shuffle=do_train)
    resize_height = 224
    resize_width = 224
    buffer_size = 1000

    # define map operations
    decode_op = C.Decode()
    resize_crop_op = C.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    resize_op = C.Resize((256, 256))
    center_crop = C.CenterCrop(resize_width)
    rescale_op = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = C.HWC2CHW()

    if do_train:
        trans = [resize_crop_op, horizontal_flip_op, rescale_op, normalize_op, change_swap_op]
    else:
        trans = [decode_op, resize_op, center_crop, normalize_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=32)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=32)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(256, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(1)

    return data_set

dataset = create_dataset_rp2k(True)


def generate_steps_lr(lr_init, steps_per_epoch, total_epochs):
    total_steps = total_epochs * steps_per_epoch
    decay_epoch_index = [0.3*total_steps, 0.6*total_steps, 0.8*total_steps]
    lr_each_step = []
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr = lr_init
        elif i < decay_epoch_index[1]:
            lr = lr_init * 0.1
        elif i < decay_epoch_index[2]:
            lr = lr_init * 0.01
        else:
            lr = lr_init * 0.001
        lr_each_step.append(lr)
    return lr_each_step

# Set epoch size
epoch_size = 60

# Wrap the backbone network with loss.
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
loss_net = nn.WithLossCell(train_network, loss_fn)
steps_per_epoch = dataset.get_dataset_size()
lr = generate_steps_lr(lr_init=0.01, steps_per_epoch=steps_per_epoch, total_epochs=epoch_size)

# Create an optimizer.
optim = Momentum(filter(lambda x: x.requires_grad, classification_layer.get_parameters()), Tensor(lr, mindspore.float32), 0.9, 4e-5)
train_net = nn.TrainOneStepCell(loss_net, optim)


for epoch in range(epoch_size):
    for i, items in enumerate(dataset):
        data, label = items
        data = mindspore.Tensor(data)
        label = mindspore.Tensor(label)

        loss = train_net(data, label)
        print(f"epoch: {epoch}/{epoch_size}, loss: {loss}")
    # Save the ckpt file for each epoch.
    ckpt_path =f"mobilenet_finetune_epoch{epoch}.ckpt"
    save_checkpoint(train_network, ckpt_path)




