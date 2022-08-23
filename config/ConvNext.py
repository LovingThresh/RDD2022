import os
import shutil

classes = os.listdir(r'F:\dataset\train-set')

_base_ = [
    'F:/RDD2022/RDD2022/cl_config/_base_/models/convnext/convnext-base.py',
    'F:/RDD2022/RDD2022/cl_config/_base_/datasets/imagenet_bs64_swin_224.py',
    'F:/RDD2022/RDD2022/cl_config/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'F:/RDD2022/RDD2022/cl_config/_base_/default_runtime.py',
]

model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=27,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)
    ))
dataset_type = 'SingleViewDataset'
data = dict(
    samples_per_gpu=48,
    workers_per_gpu=1,
    train=dict(
        data_prefix='F:/dataset/train-set',
        classes = classes,
        ann_file=None),
    val=dict(
        data_prefix='F:/dataset/val-set',
        classes = classes,
        ann_file=None)
)


optimizer = dict(lr=4e-3)
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

