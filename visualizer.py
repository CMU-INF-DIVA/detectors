import numpy as np
import torch
from detectron2.utils.visualizer import Visualizer as dt_visualizer
from matplotlib import pyplot as plt

from .base import Detection, ObjectType
from .color import ColorManager


class Visualizer(object):

    def __init__(self, object_types=ObjectType):
        self.color_manager = ColorManager()
        self.object_types = object_types

    def draw(self, image: torch.Tensor, detection: Detection,
             show: bool = True, **show_args):
        '''
        image: a pytorch float tenor as H x W x C[BGR] in [0, 256)
        detection: output from Detector
        '''
        image_rgb = image.numpy()[:, :, ::-1]
        visualizer = dt_visualizer(image_rgb, None)
        detection = detection.to('cpu')
        self._draw_detection(visualizer, detection, image_rgb)
        output = visualizer.get_output()
        visual_image = output.get_image()
        plt.close(output.fig)
        if show:
            self.plt_imshow(visual_image, **show_args)
        return visual_image

    def plt_imshow(self, image, figsize=(16, 9), dpi=120, axis='off'):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.axis(axis)
        plt.imshow(image)
        plt.show()
        plt.close(fig)

    def _draw_detection(self, visualizer, detection, image_rgb):
        labels, colors, masks = [], [], []
        for obj_i in range(len(detection)):
            obj_type = self.object_types(
                detection.object_types[obj_i].item()).name
            obj_id = obj_i
            if detection.has('track_ids'):
                obj_id = detection.track_ids[obj_i].item()
            score = detection.detection_scores[obj_i] * 100
            label = '%s-%s %.0f%%' % (obj_type, obj_id, score)
            labels.append(label)
            x0, y0 = detection.image_boxes[obj_i, :2].type(torch.int)
            x1, y1 = detection.image_boxes[obj_i, 2:].ceil().type(torch.int)
            roi = image_rgb[y0:y1, x0:x1]
            color = self.color_manager.get_color((obj_type, obj_id), roi)
            colors.append(color)
            mask = [np.array([0, 0])]
            if detection.has('image_masks'):
                mask = detection.image_masks[obj_i].numpy()
            masks.append(mask)
        if detection.has('image_locations'):
            keypoints = np.empty((len(detection), 1, 3))
            keypoints[:, 0, :2] = detection.image_locations
            keypoints[:, 0, 2] = 1
        else:
            keypoints = None
        visualizer.overlay_instances(
            masks=masks, boxes=detection.image_boxes, labels=labels,
            keypoints=keypoints, assigned_colors=colors)
        return colors
