model = dict(
    backbone=dict(
        arch='P6',
        type='YOLOv8CSPDarknetMySelf',
        out_indices = (2, 3, 4, 5),
        last_stage_out_channels = 1024
    ),
    neck = dict(
        in_channels=[
            256,
            512,
            768,
            1024,
        ],
        out_channels=[
            256,
            512,
            768,
            1024,
        ],
    ),
 
    bbox_head = dict(
           prior_generator=dict(
        strides=[
        8,
        16,
        32,
        64
    ]),
        head_module = dict(
            featmap_strides=[
                8,
                16,
                32,
                64
            ],
            in_channels=[
                256,
                512,
                768,
                1024
            ],
        ),
    )
)



