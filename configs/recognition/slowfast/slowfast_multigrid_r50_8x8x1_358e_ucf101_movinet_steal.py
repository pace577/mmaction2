model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False) ),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=600,
        spatial_type='avg',
        dropout_ratio=0.5,
        loss_cls=dict(type='CrossEntropyLoss', logit_labels=True)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

train_cfg = None
test_cfg = dict(average_clips='prob')

split = 1
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos'
data_root_val = 'data/ucf101/videos'
ann_file_train = f'data/ucf101/ucf101_train_videos.txt'
#ann_file_train = f'data/ucf101/ucf101_val_split_1_videos.txt'
ann_file_val = f'data/ucf101/ucf101_val_videos.txt'
ann_file_test = f'data/ucf101/ucf101_val_videos.txt'

# Give probabilities as labels instead of class ids
prob_labels = True
prob_labels_file_train = f'results/movinet-blackbox-labels-train.pkl'

#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)

#train_pipeline = [
#    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
#    dict(type='RawFrameDecode'),
#    dict(type='Resize', scale=(-1, 256)),
#    dict(type='RandomResizedCrop'),
#    dict(type='Resize', scale=(224, 224), keep_ratio=False),
#    dict(type='Flip', flip_ratio=0.5),
#    dict(type='Normalize', **img_norm_cfg),
#    dict(type='FormatShape', input_format='NCTHW'),
#    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#    dict(type='ToTensor', keys=['imgs', 'label'])
#]
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=3, num_clips=1),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(-1, 256)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    #dict(type='RandomCrop', size=112),
    #dict(type='Flip', flip_ratio=0.5),
    #dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
#val_pipeline = [
#    dict(
#        type='SampleFrames',
#        clip_len=32,
#        frame_interval=2,
#        num_clips=1,
#        test_mode=True),
#    dict(type='RawFrameDecode'),
#    dict(type='Resize', scale=(-1, 256)),
#    dict(type='CenterCrop', crop_size=224),
#    dict(type='Normalize', **img_norm_cfg),
#    dict(type='FormatShape', input_format='NCTHW'),
#    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#    dict(type='ToTensor', keys=['imgs'])
#]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=3,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, 128)),
    #dict(type='CenterCrop', crop_size=112),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
#test_pipeline = [
#    dict(
#        type='SampleFrames',
#        clip_len=32,
#        frame_interval=2,
#        num_clips=10,
#        test_mode=True),
#    dict(type='RawFrameDecode'),
#    dict(type='Resize', scale=(-1, 256)),
#    dict(type='ThreeCrop', crop_size=256),
#    dict(type='Normalize', **img_norm_cfg),
#    dict(type='FormatShape', input_format='NCTHW'),
#    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#    dict(type='ToTensor', keys=['imgs'])
#]

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=3,
        #num_clips=10,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, 128)),
    #dict(type='CenterCrop', crop_size=112),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        probability_labels=prob_labels,
        probability_labels_file=prob_labels_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
#optimizer = dict(type='AdamW',
#                 lr=0.0001,
#                 betas=(0.9, 0.9999),
#                 weight_decay=0.05)
optimizer = dict(
    type='SGD', lr=0.0002, momentum=0.9, 
    weight_decay=0.0001)  # 16gpu 0.1->0.2
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
#optimizer_config = dict(grad_clip=None)
#lr_config = dict(policy='step',
#                 step=1,
#                 min_lr=0.000001)
#lr_config = dict(policy='step',
#                 step=[94, 154, 196],
#                 warmup='linear',
#                 warmup_by_epoch=True,
#                 warmup_iters=20,
#                 warmup_ratio=0.02)
lr_config = dict(policy='step',
                 step=[20,40,60,80],
                 gamma=0.1,
                 min_lr=0.000002)

#optimizer_config = dict(grad_clip=None)
#lr_config = dict(
#    policy='CosineAnnealing',
#    min_lr=0,
#    warmup='linear',
#    warmup_by_epoch=True,
#    warmup_iters=2,
#    warmup_ratio=0.01)

total_epochs = 100
#total_epochs = 240
#total_epochs = 358

evaluation = dict(
    interval=3, 
    metrics=['top_k_accuracy', 'mean_class_accuracy']
    #metrics=['top_k_accuracy']
)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        #    dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'

checkpoint_config = dict(interval=5)
workflow = [('train', 1)]

find_unused_parameters = False

multigrid = dict(
    long_cycle=True,
    short_cycle=True,
    epoch_factor=1.5,
    long_cycle_factors=[[0.25, 0.7071], [0.5, 0.7071], [0.5, 1], [1, 1]],
    short_cycle_factors=[0.5, 0.7071],
    default_s=(224, 224),
)

precise_bn = dict(num_iters=200, interval=3)
#precise_bn = dict(num_iters=150, interval=3)

load_from = None
resume_from = None

work_dir = './work_dirs/slowfast_multigrid_r50_32x3x1_240e_movinet_steal'
#gpu_ids = ['cuda:0','cuda:1','cuda:2'] # added this to prevent error during training "AttributeError: 'ConfigDict' object has no attribute 'gpu_ids'"
gpu_ids = ['cuda:0'] # added this to prevent error during training "AttributeError: 'ConfigDict' object has no attribute 'gpu_ids'"
