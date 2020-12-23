import numpy as np
import torch
from detectron2.utils.visualizer import Visualizer as dt_visualizer, GenericMask
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

        return: a numpy uint8 array as H x W x C[RGB]
        '''
        image_rgb = image.numpy()[:, :, ::-1]
        visualizer = dt_visualizer(image_rgb)
        detection = detection.to('cpu')
        self._draw_detection(visualizer, detection)
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
        diagonal = np.sqrt(height * width)
        edges = detection.image_boxes[:, 2:] - detection.image_boxes[:, :2]
        areas = edges[:, 0] * edges[:, 1]
        indices = areas.argsort(descending=True)
        for idx in indices:
            obj_type = self.object_types(
                detection.object_types[idx].item()).name
            score = detection.detection_scores[idx] * 100
            bbox = detection.image_boxes[idx]
            x0, y0 = bbox[:2].type(torch.int)
            x1, y1 = bbox[2:].ceil().type(torch.int)
            roi = visualizer.img[y0:y1, x0:x1]
            if detection.has('track_ids'):
                obj_id = detection.track_ids[idx].item()
                label = '%s-%s %.0f%%' % (obj_type, obj_id, score)
                color = self.color_manager.get_color((obj_type, obj_id), roi)
            else:
                label = '%s %.0f%%' % (obj_type, score)
                color = self.color_manager.get_color(obj_type, roi)
            if detection.has('custom_labels'):
                label += detection.custom_labels[idx]
            visualizer.draw_box(bbox, edge_color=color)
            self._draw_label(
                visualizer, bbox, label, (y1 - y0) / diagonal, color)
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
    def _draw_label(visualizer, bbox, text, size_ratio, color):
        x0, y0, x1, _ = bbox
        linewidth = max(
            visualizer._default_font_size / 4, 1) * visualizer.output.scale
        position = (x0 + x1) / 2, y0 - linewidth
        font_size = visualizer._default_font_size * np.clip(
            0.4 + size_ratio * 6, 0.5, 1) * visualizer.output.scale
        box_color = (0, 0, 0) if sum(color) > 1.5 else (1, 1, 1)
        visualizer.output.ax.text(
            *position, text, size=font_size, family='sans-serif',
            bbox={'facecolor': box_color, 'pad': 0.7,
                  'edgecolor': 'none', 'alpha': 0.7},
            verticalalignment='bottom', horizontalalignment='center',
            color=color, zorder=10)
