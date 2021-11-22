import mindspore_hub as mshub
from mindspore.train.model import Model
from mindspore import context, Tensor, nn
from mindspore import save_checkpoint, load_checkpoint,load_param_into_net
from mindspore import ops
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C
from mindspore.common import dtype as mstype
from mindspore.dataset.transforms import py_transforms

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
classification_layer.set_train(False)

softmax = nn.Softmax()
network = nn.SequentialCell([network, reducemean_flatten, classification_layer, softmax])


# Load a pre-trained ckpt file.
ckpt_path = "./ckpt/mobilenet59_finetune_epoch59.ckpt"
trained_ckpt = load_checkpoint(ckpt_path)
load_param_into_net(classification_layer, trained_ckpt)

loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# Define loss and create model.
eval_dataset = create_dataset_rp2k(do_train=False)
eval_metrics = {'Loss': nn.Loss(),
                 'Top1-Acc': nn.Top1CategoricalAccuracy(),
                 'Top5-Acc': nn.Top5CategoricalAccuracy()}

model = Model(network, loss_fn=loss, optimizer=None, metrics=eval_metrics)
metrics = model.eval(eval_dataset)
print("metric: ", metrics)