import logging
import os
import os.path as osp
import sys

import torch.nn.functional as F
from torchvision.ops import boxes as box_ops


class FvcoreCachePath(object):

    def __enter__(self):
        self.cache_dir = os.environ.get('FVCORE_CACHE')
        os.environ['FVCORE_CACHE'] = osp.join(
            osp.dirname(__file__), 'weights/fvcore')

    def __exit__(self, type, value, traceback):
        if self.cache_dir is None:
            os.environ.pop('FVCORE_CACHE')
        else:
            os.environ['FVCORE_CACHE'] = self.cache_dir
        del self.cache_dir


class SysPath(object):

    def __init__(self, relative_path):
        self.path = osp.join(osp.dirname(__file__), relative_path)

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, type, value, traceback):
        sys.path.pop(0)


class MuteLogger(object):

    def __init__(self, name, level=logging.ERROR):
        self.name = name
        self.level = level

    def __enter__(self):
        self.logger = logging.getLogger(self.name)
        self.orig_level = self.logger.level
        self.logger.setLevel(self.level)

    def __exit__(self, type, value, traceback):
        self.logger.setLevel(self.orig_level)


def resize_with_padding(image, target_shape):
    '''
    image: pytorch tensor C x H x W
    target_shape: (h, w)
    '''
    target_height, target_width = target_shape
    scale_factor = min(target_height / image.shape[-2],
                       target_width / image.shape[-1])
    image = F.interpolate(
        image.unsqueeze(0), scale_factor=scale_factor,
        mode='bilinear', align_corners=False, recompute_scale_factor=False)
    pad = (target_height - image.shape[-2], target_width - image.shape[-1])
    pad = (int(round(pad[1] / 2 - 0.5)), int(round(pad[1] / 2 + 0.5)),
           int(round(pad[0] / 2 - 0.5)), int(round(pad[0] / 2 + 0.5)))
    image = F.pad(image.squeeze(0), pad)
    return image


def invert_resize_padding(boxes, current_shape, target_shape):
    '''
    boxes: pytorch tensor N x 4
    current_shape: (h, w)
    target_shape: (h, w)
    '''
    current_height, current_width = current_shape
    target_height, target_width = target_shape
    scale = max(current_shape) / max(target_shape)
    pad = ((current_width - target_width * scale) / 2,
           (current_height - target_height * scale) / 2)
    boxes[:, ::2] -= pad[0]
    boxes[:, 1::2] -= pad[1]
    boxes /= scale
    boxes = box_ops.clip_boxes_to_image(boxes, target_shape)
    return boxes
