from enum import IntEnum, auto
from typing import List, Union

import torch
from torch.backends import cudnn
from detectron2.structures import Instances

Detection = Instances
'''
Detection attributes: object_types, image_boxes, detection_scores, \
    [image_features, image_masks]
'''


class ObjectType(IntEnum):

    Vehicle = auto()
    Person = auto()
    Bike = auto()


class Detector(object):

    def __init__(self, gpu_id: Union[None, int] = None):
        self.device = 'cpu'
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = 'cuda:%d' % (gpu_id)
            cudnn.fastest = True
            cudnn.benchmark = True

    def __call__(self, images: List[torch.Tensor],
                 to_cpu: bool = True) -> List[Detection]:
        '''
        images: a list of pytorch tensors as H x W x C[BGR] in [0, 256) on cpu
        to_cpu: whether to move detections to cpu
        '''
        raise NotImplementedError

    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.device)
