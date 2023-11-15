model = dict(
    test_cfg = dict(
        nms=dict(iou_threshold=0.5, type='soft-nms'),
    )
)