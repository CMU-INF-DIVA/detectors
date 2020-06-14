import matplotlib as mpl
import numpy as np
import torch
from detectron2.utils.visualizer import Visualizer as dt_visualizer
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle

from .base import Detection, ObjectType


class Visualizer(object):

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
            self._plt_imshow(visual_image, **show_args)
        return visual_image

    def _plt_imshow(self, image, figsize=(16, 9), dpi=120, axis='off'):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.axis(axis)
        plt.imshow(image)
        plt.show()
        plt.close(fig)

    def _draw_detection(self, visualizer, detection, image_rgb):
        labels, colors, masks = [], [], []
        for obj_i in range(len(detection)):
            obj_type = ObjectType(detection.pred_classes[obj_i].item())
            obj_id = obj_i
            obj_type = obj_type.name
            score = detection.scores[obj_i] * 100
            label = '%s-%s %.0f%%' % (obj_type, obj_id, score)
            labels.append(label)
            mask = [np.array([0, 0])]
            if detection.has('pred_masks'):
                mask = detection.pred_masks[obj_i].numpy()
            masks.append(mask)
        visualizer.overlay_instances(
            masks=masks, boxes=detection.pred_boxes, labels=labels)
        return colors
