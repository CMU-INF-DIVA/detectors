import os
import os.path as osp

import torch
from detectron2.layers import batched_nms
from torchvision.ops import boxes as box_ops

from .base import Detection, Detector, ObjectType
from .utils import invert_resize_padding, resize_with_padding


class YOLOv5(Detector):

    MODEL_LIST = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']

    INPUT_SHAPES = [416, 640, 1280, 1920]

    TYPE_MAPPING = {
        0: ObjectType.Person, 2: ObjectType.Vehicle,
        5: ObjectType.Vehicle, 7: ObjectType.Vehicle,
        3: ObjectType.Vehicle, 1: ObjectType.Bike
    }

    def __init__(self, gpu_id=None,
                 model='yolov5x',
                 input_shape=640,
                 score_threshold=0.4,
                 nms_threshold=0.5,
                 interclass_nms_threshold=None):
        assert model in self.MODEL_LIST, 'Unsupported model %s' % (model)
        assert input_shape in self.INPUT_SHAPES, \
            'Invalid input shape %d' % (input_shape)
        super(YOLOv5, self).__init__(gpu_id)
        cwd = os.getcwd()
        os.chdir(osp.join(osp.dirname(__file__), 'weights/yolov5'))
        model = torch.hub.load(
            'ultralytics/yolov5', model, pretrained=True)
        os.chdir(cwd)
        model.requires_grad_(False)
        model.eval()
        self.model = model.to(self.device)
        self.selected_classes = torch.as_tensor(
            [*self.TYPE_MAPPING.keys()], dtype=torch.long,
            device=self.device).unsqueeze(0)
        self.model_input_shape = (input_shape, input_shape)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.interclass_nms_threshold = interclass_nms_threshold

    def preprocess(self, images):
        processed_images = []
        for image in images:
            image = image.to(device=self.device, non_blocking=True)
            image = image.flip(-1).permute(2, 0, 1).type(torch.float) / 255.0
            image = resize_with_padding(image, self.model_input_shape)
            processed_images.append(image)
        image_tensor = torch.stack(processed_images, dim=0)
        return image_tensor

    def inference(self, image_tensor):
        outputs = self.model(image_tensor)
        return outputs[0]

    def postprocess(self, outputs, images, to_cpu):
        detections = []
        for output, image in zip(outputs, images):
            height, width, _ = image.shape
            output = output[output[..., 4] >= self.score_threshold]
            output[:, 5:] *= output[:, 4:5]
            if output.shape[0] > 0:
                scores, types = output[:, 5:].max(dim=1)
            else:
                scores = torch.zeros_like(output[:, 0])
                types = torch.zeros_like(output[:, 0], dtype=torch.long)
            detection = Detection(
                (height, width), object_types=types,
                image_boxes=output[:, :4], detection_scores=scores)
            keep = (types.unsqueeze(-1) == self.selected_classes).any(dim=-1)
            detection = detection[keep]
            detection.object_types = torch.as_tensor(
                [self.TYPE_MAPPING[object_type.item()]
                 for object_type in detection.object_types],
                device=detection.object_types.device)
            boxes = torch.empty_like(detection.image_boxes)
            xy = detection.image_boxes[:, :2]
            wh_half = detection.image_boxes[:, 2:] / 2
            boxes[:, :2] = xy - wh_half
            boxes[:, 2:] = xy + wh_half
            detection.image_boxes = boxes
            keep = batched_nms(
                boxes, detection.detection_scores, detection.object_types,
                self.nms_threshold)
            detection = detection[keep]
            if self.interclass_nms_threshold is not None and len(detection) > 0:
                keep = box_ops.nms(
                    detection.image_boxes, detection.detection_scores,
                    self.interclass_nms_threshold)
                detection = detection[keep]
            detection.image_boxes = invert_resize_padding(
                detection.image_boxes, self.model_input_shape, (height, width))
            if to_cpu:
                detection = detection.to('cpu')
            detections.append(detection)
        return detections

    def __call__(self, images, to_cpu=True):
        with torch.no_grad():
            image_tensor = self.preprocess(images)
            outputs = self.inference(image_tensor)
            detections = self.postprocess(outputs, images, to_cpu)
        return detections
