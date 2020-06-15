from enum import IntEnum
from typing import List, Union

import torch
from detectron2.structures import Instances

Detection = Instances
'''
Detection attributes: object_types, image_boxes, detection_scores, \
    [image_features, image_masks]
'''

ObjectType = IntEnum('ObjectType', ['Vehicle', 'Person', 'Bike'])


class Detector(object):

    def __init__(self, gpu_id: Union[None, int] = None):
        self.device = 'cpu'
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = 'cuda:%d' % (gpu_id)

    def __call__(self, images: List[torch.Tensor],
                 to_cpu: bool = True) -> List[Detection]:
        '''
        images: a list of pytorch tensors as H x W x C[BGR] in [0, 256)
        to_cpu: whether to move detections to cpu
        '''
        raise NotImplementedError

    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.device)
