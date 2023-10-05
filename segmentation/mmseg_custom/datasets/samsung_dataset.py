# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class SamsungDataset(CustomDataset):
    """Samsung dataset.
    """

    CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12')

    # CLASSES = ('Background', 'Road', 'Sidewalk', 'Construction', 'Fence', 'Pole', 'Traffic Light',
    #            'Traffic Sign', 'Nature', 'Sky', 'Person', 'Rider', 'Car')

    PALETTE = [[0, 0, 0],
               [165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
               [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
               [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96]]

    def __init__(self, **kwargs):
        super(SamsungDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)