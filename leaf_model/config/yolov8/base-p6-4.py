model = dict(
    backbone=dict(
        arch='P6',
        type='YOLOv8CSPDarknet',
        out_indices = (2, 3, 4, 5),
        last_stage_out_channels = 1024
    ),
    neck = dict(
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        out_channels=[
            128,
            256,
            512,
            1024,
        ],
    ),
 
    bbox_head = dict(
           prior_generator=dict(
        strides=[
        4,
        8,
        16,
        32,
    ]),
        head_module = dict(
            featmap_strides=[
                4,      
                8,
                16,
                32,
            ],
            in_channels=[
                128,
                256,
                512,
                1024,
            ],
        ),
    )
)



