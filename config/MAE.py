# -*- coding: utf-8 -*-
# @Time    : 2022/8/20 21:14
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : MAE.py
# @Software: PyCharm

# mixed precision
fp16 = dict(loss_scale='dynamic')

model = dict(

    type='MAE',
    backbone=dict(type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75,
                  init_cfg=dict(
                      type='Pretrained',
                      checkpoint='mae_vit-base-p16_8xb512-coslr-400e_in1k-224_20220223-85be947b.pth',
                  )),
    neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(type='MAEPretrainHead', norm_pix=True, patch_size=16))

# dataset

# dataset settings
data_source = 'ImageNet'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.2, 1.0), interpolation=3),
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='F:/dataset/train-set',
            ann_file=None,
        ),
        pipeline=train_pipeline,
        prefetch=prefetch))

# optimizer
cudnn_benchmark = True  # 是否是使用 cudnn_benchmark 去加速，它对于固定输入大小的可以提高训练速度。
# optimizer

optimizer = dict(
    type='AdamW',
    lr=1.5e-4 * 4096 / 256,
    betas=(0.9, 0.95),
    weight_decay=0.05,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.)
    })
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=0.0,
    warmup='linear',
    warmup_iters=40,
    warmup_ratio=1e-4,
    warmup_by_epoch=True,
    by_epoch=False)

# schedule
runner = dict(type='EpochBasedRunner', max_epochs=10)

# runtime
checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='')
persistent_workers = True
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])

# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
