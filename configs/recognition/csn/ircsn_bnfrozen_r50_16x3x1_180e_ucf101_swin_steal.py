_base_ = [
    '../../_base_/models/ircsn_r152.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dCSN',
        pretrained2d=False,
        pretrained=None,
        #pretrained=  # noqa: E251
        #'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth'  # noqa: E501
        depth=50,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=False,
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        loss_cls=dict(type='CrossEntropyLoss', logit_labels=False)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob', max_testing_views=10))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos'
data_root_val = 'data/ucf101/videos'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'data/ucf101/ucf101_train_videos.txt'
#ann_file_train = f'data/ucf101/ucf101_train_split_{split}_videos.txt'
ann_file_val = f'data/ucf101/ucf101_val_split_{split}_videos.txt'
ann_file_test = f'data/ucf101/ucf101_val_split_{split}_videos.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)

# Give probabilities as labels instead of class ids
prob_labels = True
prob_labels_file_train = f'results/swin-blackbox-labels-train.pkl'

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=3, num_clips=1),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(128, 171)),
    #dict(type='RandomCrop', size=112),
    #dict(type='Flip', flip_ratio=0.5),
    dict(type='Resize', scale=(128,128), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=3,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Resize', scale=(128,128), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Resize', scale=(128,128), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=30,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
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
optimizer = dict(
    type='SGD', lr=0.08, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=40)
total_epochs = 180

checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, 
    #metrics=['top_k_accuracy', 'mean_class_accuracy']
    metrics=[]
)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ircsn_bnfrozen_r50_16x3x1_180e_ucf101_swin_steal'  # noqa: E501
load_from = None
resume_from = None
workflow = [('train', 1)]
