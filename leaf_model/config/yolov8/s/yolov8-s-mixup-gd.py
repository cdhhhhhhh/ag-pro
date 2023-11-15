_base_ = [
    '/home/neau/sdb/ag-pro/leaf_model/config/yolov8/yolov8-s-ag-mixup.py',
]



model = dict(
    neck=dict(
        _delete_ = True,
        num_repeats=[12, 12, 12, 12,12, 12],
        channels_list=[0, 0, 0, 128, 256, 512, 1024, 512, 512, 512],
        extra_cfg=dict(
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                depths=2,
                fusion_in=960,
                ppa_in=704,
                fusion_act=dict(type='ReLU6'),
                fuse_block_num=3,
                embed_dim_p=128,
                embed_dim_n=704,
                key_dim=8,
                num_heads=4,
                mlp_ratios=1,
                attn_ratios=2,
                c2t_stride=2,
                drop_path_rate=0.1,
                trans_channels=[128, 64, 128, 256],
                pool_mode='torch'
        ),
        type='GDSNeck',
        ),
)