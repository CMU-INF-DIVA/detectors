__author__ = 'Lijun Yu'

from .base import ObjectType
from .color import ColorManager
from .visualizer import Visualizer


def get_detector(name):
    if name == 'Mask R-CNN':
        from .mask_rcnn import MaskRCNN
        return MaskRCNN
    elif name == 'YOLOv5':
        from .yolo_v5 import YOLOv5
        return YOLOv5
    elif name == 'Efficient Det':
        from .efficient_det import EfficientDet
        return EfficientDet
    else:
        raise NotImplementedError('Detector<%s> not found' % (name))
