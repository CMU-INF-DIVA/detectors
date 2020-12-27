from typing import Union

import numpy as np
import torch
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import Visualizer as dt_visualizer
from matplotlib import pyplot as plt

from .base import Detection, ObjectType
from .color import ColorManager


class Visualizer(object):

    def __init__(self, object_types=ObjectType):
        self.color_manager = ColorManager()
        self.object_types = object_types

    def draw(self, image: Union[torch.Tensor, np.array], detection: Detection,
             image_id: str = None, *, show: bool = True, **show_args):
        '''
        image: a pytorch float tensor as H x W x C[BGR] in [0, 256) or
            a numpy uint8 array as H x W x C[RGB]
        detection: output from Detector

        return: a numpy uint8 array as H x W x C[RGB]
        '''
        if isinstance(image, torch.Tensor):
            image = image.numpy()[:, :, ::-1]
        visualizer = dt_visualizer(image)
        detection = detection.to('cpu')
        self._draw_detection(visualizer, detection)
        if image_id is not None:
            visualizer.draw_text(image_id, (0, 0), horizontal_alignment='left')
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

    def _draw_detection(self, visualizer, detection):
        height, width = visualizer.img.shape[:2]
        edges = detection.image_boxes[:, 2:] - detection.image_boxes[:, :2]
        areas = edges[:, 0] * edges[:, 1]
        indices = areas.argsort(descending=True)
        use_existing_color = False
        if detection.has('colors'):
            use_existing_color = True
        elif detection.has('track_ids'):
            detection.colors = torch.empty((len(detection), 3))
        for idx in indices:
            obj_type = self.object_types(
                detection.object_types[idx].item()).name
            score = detection.detection_scores[idx] * 100
            bbox = detection.image_boxes[idx]
            if detection.has('track_ids'):
                obj_id = detection.track_ids[idx].item()
                label = '%s-%s %.0f%%' % (obj_type, obj_id, score)
            else:
                obj_id = None
                label = '%s %.0f%%' % (obj_type, score)
            if detection.has('custom_labels'):
                label += detection.custom_labels[idx]
            if use_existing_color:
                color = detection.colors[idx].numpy()
            else:
                if detection.has('track_ids'):
                    color = self.color_manager.get_color(
                        obj_id, visualizer.img, bbox.type(torch.int))
                    detection.colors[idx] = torch.as_tensor(color)
                else:
                    color = self.color_manager.get_color(obj_type)
            visualizer.draw_box(bbox, edge_color=color)
            self._draw_label(visualizer, bbox, label, color)
            if detection.has('image_masks'):
                mask = detection.image_masks[idx].numpy()
                mask = GenericMask(mask, height, width)
                for segment in mask.polygons:
                    visualizer.draw_polygon(segment.reshape(-1, 2), color)
            if detection.has('image_locations'):
                keypoints = np.empty((len(detection), 1, 3))
                keypoints[:, 0, :2] = detection.image_locations
                keypoints[:, 0, 2] = 1

    @staticmethod
    def _draw_label(visualizer, bbox, text, color):
        x0, y0, x1, y1 = bbox
        linewidth = max(
            visualizer._default_font_size / 4, 1) * visualizer.output.scale
        position = (x0 + x1) / 2, y0 - linewidth
        size_ratio = (y1 - y0) / visualizer.output.height
        font_size = visualizer._default_font_size * np.clip(
            0.4 + size_ratio * 5, 0.5, 1) * visualizer.output.scale
        box_color = (0, 0, 0) if sum(color) > 1 else (1, 1, 1)
        visualizer.output.ax.text(
            *position, text, size=font_size, family='sans-serif',
            bbox={'facecolor': box_color, 'pad': 0.7,
                  'edgecolor': 'none', 'alpha': 0.7},
            verticalalignment='bottom', horizontalalignment='center',
            color=color, zorder=10)
