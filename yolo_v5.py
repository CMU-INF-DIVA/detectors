import os
import os.path as osp
import torch
import torch.nn.functional as F
from detectron2.layers import batched_nms
from torchvision.ops import boxes as box_ops

from .base import Detection, Detector, ObjectType

MODEL_LIST = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
DEFAULT_MODEL = 'yolov5x'

INPUT_SHAPES = [416, 640, 1280, 1920]
DEFAULT_SHAPE = 640

TYPE_MAPPING = {
    0: ObjectType.Person, 2: ObjectType.Vehicle,
    5: ObjectType.Vehicle, 7: ObjectType.Vehicle,
    3: ObjectType.Vehicle, 1: ObjectType.Bike
}


class YoloV5(Detector):

    def __init__(self, gpu_id=None, model=DEFAULT_MODEL,
                 input_shape=DEFAULT_SHAPE,
                 score_threshold=0.4, nms_threshold=0.5,
                 interclass_nms_threshold=None):
        assert input_shape in INPUT_SHAPES, \
            'Invalid input shape %d' % (input_shape)
        super(YoloV5, self).__init__(gpu_id)
        cwd = os.getcwd()
        os.chdir(osp.join(osp.dirname(__file__), 'weights/yolov5'))
        model = torch.hub.load(
            'ultralytics/yolov5', model, pretrained=True)
        os.chdir(cwd)
        self.model = model.to(self.device).eval()
        self.selected_classes = torch.as_tensor(
            [*TYPE_MAPPING.keys()], dtype=torch.long, device=self.device)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.interclass_nms_threshold = interclass_nms_threshold
        self.model_input_shape = (input_shape, input_shape)

    def preprocess(self, images):
        processed_images = []
        for image in images:
            image = image.to(device=self.device, non_blocking=True)
            image = image.flip(-1).permute(2, 0, 1).type(torch.float) / 255.0
            scale_factor = min(self.model_input_shape[-2] / image.shape[-2],
                               self.model_input_shape[-1] / image.shape[-1])
            image = F.interpolate(
                image.unsqueeze(0), scale_factor=scale_factor,
                mode='bilinear', align_corners=False,
                recompute_scale_factor=False)
            pad = (self.model_input_shape[-2] - image.shape[-2],
                   self.model_input_shape[-1] - image.shape[-1])
            pad = (int(round(pad[1] / 2 - 0.5)), int(round(pad[1] / 2 + 0.5)),
                   int(round(pad[0] / 2 - 0.5)), int(round(pad[0] / 2 + 0.5)))
            image = F.pad(image.squeeze(0), pad)
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
            keep = (types.unsqueeze(1) ==
                    self.selected_classes.unsqueeze(0)).any(dim=1)
            detection = detection[keep]
            detection.object_types = torch.as_tensor(
                [TYPE_MAPPING[object_type.item()]
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
            scale = max(self.model_input_shape) / max(image.shape[-2:])
            pad = ((self.model_input_shape[-1] - width * scale) / 2,
                   (self.model_input_shape[-2] - height * scale) / 2)
            detection.raw_boxes = detection.image_boxes
            boxes = detection.image_boxes.clone()
            boxes[:, ::2] -= pad[0]
            boxes[:, 1::2] -= pad[1]
            detection.image_boxes = box_ops.clip_boxes_to_image(
                boxes / scale, (height, width))
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
