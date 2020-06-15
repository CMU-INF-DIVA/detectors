__author__ = 'Lijun Yu'


def get(name):
    if name == 'Mask R-CNN':
        from .mask_rcnn import MaskRCNN
        model = MaskRCNN
    elif name == 'YOLOv5':
        from .yolo_v5 import YOLOv5
        model = YOLOv5
    else:
        raise NotImplementedError('%s model not found' % (name))
    return model
