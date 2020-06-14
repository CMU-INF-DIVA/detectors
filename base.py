from enum import IntEnum
from typing import List, Union

import torch
from detectron2.structures import Instances

Detection = Instances

ObjectType = IntEnum('ObjectType', ['Vehicle', 'Person', 'Bike'])


class Detector(object):

    def __init__(self, gpu_id: Union[None, int] = None):
        self.device = 'cpu'
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = 'cuda:%d' % (gpu_id)

    def detect(self, images: List[torch.Tensor],
               to_cpu=True) -> List[Detection]:
        # images: list of float tensors as H x W x C in [0, 256)
        raise NotImplementedError

    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.device)
