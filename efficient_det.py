import torch
from detectron2.layers import batched_nms
from fvcore.common.file_io import PathManager
from torchvision.ops import boxes as box_ops

from .base import Detection, Detector, ObjectType
from .utils import (FvcoreCachePath, SysPath, invert_resize_padding,
                    resize_with_padding)


class EfficientDet(Detector):

    MODEL_LIST = [f'd{i}' for i in range(8)]
    INPUT_SHAPES = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    WEIGHTS_URL = \
        'https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/'\
        'releases/download/1.0/efficientdet-d{compound_coef}.pth'

    TYPE_MAPPING = {
        0: ObjectType.Person, 2: ObjectType.Vehicle,
        5: ObjectType.Vehicle, 7: ObjectType.Vehicle,
        3: ObjectType.Vehicle, 1: ObjectType.Bike
    }

    def __init__(self, gpu_id=None,
                 model='d3',
                 score_threshold=0.2,
                 nms_threshold=0.2,
                 interclass_nms_threshold=None):
        assert model in self.MODEL_LIST, 'Unsupported model %s' % (model)
        super(EfficientDet, self).__init__(gpu_id)
        compound_coef = int(model[1:])
        with SysPath('efficientdet'):
            from backbone import EfficientDetBackbone
            from efficientdet.utils import BBoxTransform
            model = EfficientDetBackbone(
                compound_coef=compound_coef, num_classes=90)
            self.bbox_transform = BBoxTransform()
        with FvcoreCachePath():
            ckpt_path = PathManager.get_local_path(
                self.WEIGHTS_URL.format(compound_coef=compound_coef))
        model.load_state_dict(torch.load(ckpt_path))
        model.requires_grad_(False)
        model.eval()
        self.model = model.to(self.device)
        self.model_input_shape = [self.INPUT_SHAPES[compound_coef]] * 2
        self.mean = torch.as_tensor(
            [0.406, 0.456, 0.485], device=self.device).unsqueeze(1).unsqueeze(2)
        self.std = torch.as_tensor(
            [0.225, 0.224, 0.229], device=self.device).unsqueeze(1).unsqueeze(2)
        self.selected_classes = torch.as_tensor(
            [*self.TYPE_MAPPING.keys()], dtype=torch.long,
            device=self.device).unsqueeze(0).unsqueeze(0)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.interclass_nms_threshold = interclass_nms_threshold

    def preprocess(self, images):
        processed_images = []
        for image in images:
            image = image.to(device=self.device, non_blocking=True)
            image = image.flip(-1).permute(2, 0, 1).type(torch.float) / 255.0
            image = (image - self.mean) / self.std
            image = resize_with_padding(image, self.model_input_shape)
            processed_images.append(image)
        image_tensor = torch.stack(processed_images, dim=0)
        return image_tensor

    def inference(self, image_tensor):
        features, regression, classification, anchors = self.model(
            image_tensor)
        boxes = self.bbox_transform(anchors, regression)
        boxes = box_ops.clip_boxes_to_image(boxes, image_tensor.shape[-2:])
        output = (classification, boxes, features)
        return output

    def postprocess(self, output, images, to_cpu):
        classification_batch, boxes_batch, _ = output
        scores_batch, classes_batch = classification_batch.max(dim=2)
        keep_batch = (scores_batch > self.score_threshold) & (
            classes_batch.unsqueeze(-1) == self.selected_classes).any(dim=-1)
        detections = []
        for image_i, image in enumerate(images):
            height, width, _ = image.shape
            keep = keep_batch[image_i]
            boxes = boxes_batch[image_i, keep]
            scores = scores_batch[image_i, keep]
            classes = classes_batch[image_i, keep]
            object_types = torch.as_tensor(
                [self.TYPE_MAPPING[object_type.item()]
                 for object_type in classes], device=classes.device)
            detection = Detection(
                (height, width), object_types=object_types,
                image_boxes=boxes, detection_scores=scores)
            keep = batched_nms(boxes, scores, object_types, self.nms_threshold)
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
            output = self.inference(image_tensor)
            detections = self.postprocess(output, images, to_cpu)
        return detections
