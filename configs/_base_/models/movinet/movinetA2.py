"""
param list for MoViNetA2:
-------------------------------------------------------------------------------------------------------------
|  name  | input_channels | out_channels | expanded_channels | kernel_size | stride | padding | padding_avg | 
-------------------------------------------------------------------------------------------------------------
| conv1  |       3        |      16      |        -          |   (1,3,3)   | (1,2,2)| (0,1,1) |       -     |
-------------------------------------------------------------------------------------------------------------
|Block2_1|      16        |      16      |        40         |   (1,5,5)   | (1,2,2)| (0,1,1) |   (0,0,0)   |
-------------------------------------------------------------------------------------------------------------
|Block2_2|      16        |      16      |        40         |   (3,3,3)   | (1,1,1)| (1,1,1) |   (0,1,1)   |
-------------------------------------------------------------------------------------------------------------
|Block2_3|      16        |      16      |        64         |   (3,3,3)   | (1,2,2)| (0,1,1) |   (0,0,0)   |
-------------------------------------------------------------------------------------------------------------

"""

cfg = dict()
cfg['name'] = 'A2'
cfg['conv1'] = dict()
cfg['conv1']['input_channels'] = 3
cfg['conv1']['out_channels'] = 16
cfg['conv1']['kernel_size'] = (1,3,3)
cfg['conv1']['stride'] = (1,2,2)
cfg['conv1']['padding'] = (0,1,1)
# fill_conv(cfg['conv1'], 3, 24, (1,3,3), (1,2,2), (0,1,1))
cfg['blocks'] = [[dict() for _ in range(3)],
                 [dict() for _ in range(5)],
                 [dict() for _ in range(5)],
                 [dict() for _ in range(6)],
                 [dict() for _ in range(7)]]

# block 2
# cfg['blocks'][0][0], 16, 16, 40, (1,5,5), (1,2,2), (0,1,1), (0,0,0)
cfg['blocks'][0][0]['input_channels'] = 16
cfg['blocks'][0][0]['out_channels'] = 16
cfg['blocks'][0][0]['expanded_channels'] = 40
cfg['blocks'][0][0]['kernel_size'] = (1,5,5)
cfg['blocks'][0][0]['stride'] = (1,2,2)
cfg['blocks'][0][0]['padding'] = (0,1,1)
cfg['blocks'][0][0]['padding_avg'] = (0,0,0)
# cfg['blocks'][0][1], 16, 16, 40, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][0][1]['input_channels'] = 16
cfg['blocks'][0][1]['out_channels'] = 16
cfg['blocks'][0][1]['expanded_channels'] = 40
cfg['blocks'][0][1]['kernel_size'] = (3,3,3)
cfg['blocks'][0][1]['stride'] = (1,1,1)
cfg['blocks'][0][1]['padding'] = (1,1,1)
cfg['blocks'][0][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][0][2], 16, 16, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][0][2]['input_channels'] = 16
cfg['blocks'][0][2]['out_channels'] = 16
cfg['blocks'][0][2]['expanded_channels'] = 64
cfg['blocks'][0][2]['kernel_size'] = (3,3,3)
cfg['blocks'][0][2]['stride'] = (1,1,1)
cfg['blocks'][0][2]['padding'] = (1,1,1)
cfg['blocks'][0][2]['padding_avg'] = (0,1,1)

# block 3
# cfg['blocks'][1][0], 16, 40, 96, (3,3,3), (1,2,2), (1,1,1), (0,1,1)
cfg['blocks'][1][0]['input_channels'] = 16
cfg['blocks'][1][0]['out_channels'] = 40
cfg['blocks'][1][0]['expanded_channels'] = 96
cfg['blocks'][1][0]['kernel_size'] = (3,3,3)
cfg['blocks'][1][0]['stride'] = (1,2,2)
cfg['blocks'][1][0]['padding'] = (1,1,1)
cfg['blocks'][1][0]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][1], 40, 40, 120, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][1]['input_channels'] = 40
cfg['blocks'][1][1]['out_channels'] = 40
cfg['blocks'][1][1]['expanded_channels'] = 120
cfg['blocks'][1][1]['kernel_size'] = (3,3,3)
cfg['blocks'][1][1]['stride'] = (1,1,1)
cfg['blocks'][1][1]['padding'] = (1,1,1)
cfg['blocks'][1][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][2], 40, 40, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][2]['input_channels'] = 40
cfg['blocks'][1][2]['out_channels'] = 40
cfg['blocks'][1][2]['expanded_channels'] = 96
cfg['blocks'][1][2]['kernel_size'] = (3,3,3)
cfg['blocks'][1][2]['stride'] = (1,1,1)
cfg['blocks'][1][2]['padding'] = (1,1,1)
cfg['blocks'][1][2]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][3], 40, 40, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][3]['input_channels'] = 40
cfg['blocks'][1][3]['out_channels'] = 40
cfg['blocks'][1][3]['expanded_channels'] = 96
cfg['blocks'][1][3]['kernel_size'] = (3,3,3)
cfg['blocks'][1][3]['stride'] = (1,1,1)
cfg['blocks'][1][3]['padding'] = (1,1,1)
cfg['blocks'][1][3]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][4], 40, 40, 120, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][4]['input_channels'] = 40
cfg['blocks'][1][4]['out_channels'] = 40
cfg['blocks'][1][4]['expanded_channels'] = 120
cfg['blocks'][1][4]['kernel_size'] = (3,3,3)
cfg['blocks'][1][4]['stride'] = (1,1,1)
cfg['blocks'][1][4]['padding'] = (1,1,1)
cfg['blocks'][1][4]['padding_avg'] = (0,1,1)

# block 4
# cfg['blocks'][2][0], 40, 72, 240, (5,3,3), (1,2,2), (2,1,1), (0,1,1)
cfg['blocks'][2][0]['input_channels'] = 40
cfg['blocks'][2][0]['out_channels'] = 72
cfg['blocks'][2][0]['expanded_channels'] = 240
cfg['blocks'][2][0]['kernel_size'] = (5,3,3)
cfg['blocks'][2][0]['stride'] = (1,2,2)
cfg['blocks'][2][0]['padding'] = (2,1,1)
cfg['blocks'][2][0]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][1], 72, 72, 160, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][1]['input_channels'] = 72
cfg['blocks'][2][1]['out_channels'] = 72
cfg['blocks'][2][1]['expanded_channels'] = 160
cfg['blocks'][2][1]['kernel_size'] = (3,3,3)
cfg['blocks'][2][1]['stride'] = (1,1,1)
cfg['blocks'][2][1]['padding'] = (1,1,1)
cfg['blocks'][2][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][2], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][2]['input_channels'] = 72
cfg['blocks'][2][2]['out_channels'] = 72
cfg['blocks'][2][2]['expanded_channels'] = 240
cfg['blocks'][2][2]['kernel_size'] = (3,3,3)
cfg['blocks'][2][2]['stride'] = (1,1,1)
cfg['blocks'][2][2]['padding'] = (1,1,1)
cfg['blocks'][2][2]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][3], 72, 72, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][3]['input_channels'] = 72
cfg['blocks'][2][3]['out_channels'] = 72
cfg['blocks'][2][3]['expanded_channels'] = 192
cfg['blocks'][2][3]['kernel_size'] = (3,3,3)
cfg['blocks'][2][3]['stride'] = (1,1,1)
cfg['blocks'][2][3]['padding'] = (1,1,1)
cfg['blocks'][2][3]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][4], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][4]['input_channels'] = 72
cfg['blocks'][2][4]['out_channels'] = 72
cfg['blocks'][2][4]['expanded_channels'] = 240
cfg['blocks'][2][4]['kernel_size'] = (3,3,3)
cfg['blocks'][2][4]['stride'] = (1,1,1)
cfg['blocks'][2][4]['padding'] = (1,1,1)
cfg['blocks'][2][4]['padding_avg'] = (0,1,1)

# block 5
# cfg['blocks'][3][0], 72, 72, 240, (5,3,3), (1,1,1), (2,1,1), (0,1,1)
cfg['blocks'][3][0]['input_channels'] = 72
cfg['blocks'][3][0]['out_channels'] = 72
cfg['blocks'][3][0]['expanded_channels'] = 240
cfg['blocks'][3][0]['kernel_size'] = (5,3,3)
cfg['blocks'][3][0]['stride'] = (1,1,1)
cfg['blocks'][3][0]['padding'] = (2,1,1)
cfg['blocks'][3][0]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][1], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][1]['input_channels'] = 72
cfg['blocks'][3][1]['out_channels'] = 72
cfg['blocks'][3][1]['expanded_channels'] = 240
cfg['blocks'][3][1]['kernel_size'] = (3,3,3)
cfg['blocks'][3][1]['stride'] = (1,1,1)
cfg['blocks'][3][1]['padding'] = (1,1,1)
cfg['blocks'][3][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][2], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][2]['input_channels'] = 72
cfg['blocks'][3][2]['out_channels'] = 72
cfg['blocks'][3][2]['expanded_channels'] = 240
cfg['blocks'][3][2]['kernel_size'] = (3,3,3)
cfg['blocks'][3][2]['stride'] = (1,1,1)
cfg['blocks'][3][2]['padding'] = (1,1,1)
cfg['blocks'][3][2]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][3], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][3]['input_channels'] = 72
cfg['blocks'][3][3]['out_channels'] = 72
cfg['blocks'][3][3]['expanded_channels'] = 240
cfg['blocks'][3][3]['kernel_size'] = (3,3,3)
cfg['blocks'][3][3]['stride'] = (1,1,1)
cfg['blocks'][3][3]['padding'] = (1,1,1)
cfg['blocks'][3][3]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][4], 72, 72, 144, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][3][4]['input_channels'] = 72
cfg['blocks'][3][4]['out_channels'] = 72
cfg['blocks'][3][4]['expanded_channels'] = 144
cfg['blocks'][3][4]['kernel_size'] = (1,5,5)
cfg['blocks'][3][4]['stride'] = (1,1,1)
cfg['blocks'][3][4]['padding'] = (0,2,2)
cfg['blocks'][3][4]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][5], 72, 72, 240, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][5]['input_channels'] = 72
cfg['blocks'][3][5]['out_channels'] = 72
cfg['blocks'][3][5]['expanded_channels'] = 240
cfg['blocks'][3][5]['kernel_size'] = (3,3,3)
cfg['blocks'][3][5]['stride'] = (1,1,1)
cfg['blocks'][3][5]['padding'] = (1,1,1)
cfg['blocks'][3][5]['padding_avg'] = (0,1,1)

# block 6
# cfg['blocks'][4][0], 72, 144, 480, (5,3,3), (1,2,2), (2,1,1), (0,1,1)
cfg['blocks'][4][0]['input_channels'] = 72
cfg['blocks'][4][0]['out_channels'] = 144
cfg['blocks'][4][0]['expanded_channels'] = 480
cfg['blocks'][4][0]['kernel_size'] = (5,3,3)
cfg['blocks'][4][0]['stride'] = (1,2,2)
cfg['blocks'][4][0]['padding'] = (2,1,1)
cfg['blocks'][4][0]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][1], 144, 144, 384, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][1]['input_channels'] = 144
cfg['blocks'][4][1]['out_channels'] = 144
cfg['blocks'][4][1]['expanded_channels'] = 384
cfg['blocks'][4][1]['kernel_size'] = (1,5,5)
cfg['blocks'][4][1]['stride'] = (1,1,1)
cfg['blocks'][4][1]['padding'] = (0,2,2)
cfg['blocks'][4][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][2], 144, 144, 384, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][2]['input_channels'] = 144
cfg['blocks'][4][2]['out_channels'] = 144
cfg['blocks'][4][2]['expanded_channels'] = 384
cfg['blocks'][4][2]['kernel_size'] = (1,5,5)
cfg['blocks'][4][2]['stride'] = (1,1,1)
cfg['blocks'][4][2]['padding'] = (0,2,2)
cfg['blocks'][4][2]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][3], 144, 144, 480, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][3]['input_channels'] = 144
cfg['blocks'][4][3]['out_channels'] = 144
cfg['blocks'][4][3]['expanded_channels'] = 480
cfg['blocks'][4][3]['kernel_size'] = (1,5,5)
cfg['blocks'][4][3]['stride'] = (1,1,1)
cfg['blocks'][4][3]['padding'] = (0,2,2)
cfg['blocks'][4][3]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][4], 144, 144, 480, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][4]['input_channels'] = 144
cfg['blocks'][4][4]['out_channels'] = 144
cfg['blocks'][4][4]['expanded_channels'] = 480
cfg['blocks'][4][4]['kernel_size'] = (1,5,5)
cfg['blocks'][4][4]['stride'] = (1,1,1)
cfg['blocks'][4][4]['padding'] = (0,2,2)
cfg['blocks'][4][4]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][5], 144, 144, 480, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][4][5]['input_channels'] = 144
cfg['blocks'][4][5]['out_channels'] = 144
cfg['blocks'][4][5]['expanded_channels'] = 480
cfg['blocks'][4][5]['kernel_size'] = (3,3,3)
cfg['blocks'][4][5]['stride'] = (1,1,1)
cfg['blocks'][4][5]['padding'] = (1,1,1)
cfg['blocks'][4][5]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][6], 144, 144, 576, (1,3,3), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][6]['input_channels'] = 144
cfg['blocks'][4][6]['out_channels'] = 144
cfg['blocks'][4][6]['expanded_channels'] = 576
cfg['blocks'][4][6]['kernel_size'] = (1,3,3)
cfg['blocks'][4][6]['stride'] = (1,1,1)
cfg['blocks'][4][6]['padding'] = (0,2,2)
cfg['blocks'][4][6]['padding_avg'] = (0,1,1)

# cfg['conv7'], 144, 640, (1,1,1), (1,1,1), (0,0,0)
cfg['conv7'] = dict()
cfg['conv7']['input_channels'] = 144
cfg['conv7']['out_channels'] = 640
cfg['conv7']['kernel_size'] = (1,1,1)
cfg['conv7']['stride'] = (1,1,1)
cfg['conv7']['padding'] = (0,0,0)


cfg['dense9'] = dict()
cfg['dense9']['hidden_dim'] = 2048



model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MoViNet',
        cfg=cfg,
        causal=False,
        conv_type="3d",
        tf_like=True
    ),
    cls_head=dict(
        type='MoViNetHead',
        num_classes=600,
        dropout_ratio=0.5
    ),
    train_cfg=None,
    #test_cfg=dict(maximize_clips='score')
)

