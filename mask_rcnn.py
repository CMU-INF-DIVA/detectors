import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList
from torchvision.ops.boxes import nms

from .base import Detection, Detector, ObjectType

TYPE_MAPPING = {
    'person': ObjectType.Person, 'car': ObjectType.Vehicle,
    'bus': ObjectType.Vehicle, 'truck': ObjectType.Vehicle,
    'motorcycle': ObjectType.Vehicle, 'bicycle': ObjectType.Bike
}


CFG_FILES = {
    'res50': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'res101': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
    'res101x': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
}
DEFAULT_MODEL = 'res101'


class MaskRCNN(Detector):

    def __init__(self, gpu_id=None, model=DEFAULT_MODEL,
                 input_shape=(1333, 800), score_threshold=0.5,
                 interclass_nms_threshold=None):
        super(MaskRCNN, self).__init__(gpu_id)
        cfg_file = CFG_FILES[model]
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.INPUT.MAX_SIZE_TEST = max(input_shape)
        cfg.INPUT.MIN_SIZE_TEST = min(input_shape)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
        cfg.MODEL.DEVICE = self.device
        self.model_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.cfg = cfg
        self.interclass_nms_threshold = interclass_nms_threshold
        cache_dir = os.environ.get('FVCORE_CACHE', '')
        os.environ['FVCORE_CACHE'] = osp.join(
            osp.dirname(__file__), 'weights/fvcore')
        self.predictor = DefaultPredictor(cfg)
        os.environ['FVCORE_CACHE'] = cache_dir

    def preprocess(self, images):
        processed_images = []
        for image in images:
            height, width = image.shape[:2]
            image = image.to(device=self.device, non_blocking=True)
            image = image.permute(2, 0, 1).type(torch.float)
            origin_ratio = width / height
            cfg_ratio = self.cfg.INPUT.MAX_SIZE_TEST / \
                self.cfg.INPUT.MIN_SIZE_TEST
            if cfg_ratio > origin_ratio:
                target_height = self.cfg.INPUT.MIN_SIZE_TEST
                target_width = int(round(target_height * origin_ratio))
            else:
                target_width = self.cfg.INPUT.MAX_SIZE_TEST
                target_height = int(round(target_width / origin_ratio))
            target_shape = (target_height, target_width)
            image = F.interpolate(
                image.unsqueeze(0), target_shape,
                mode='bilinear', align_corners=False)
            image = (image.squeeze(0) - self.predictor.model.pixel_mean) / \
                self.predictor.model.pixel_std
            processed_images.append(image)
        images = ImageList.from_tensors(
            processed_images, self.predictor.model.backbone.size_divisibility)
        return images

    def inference(self, images):
        model = self.predictor.model
        assert not model.training
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)
        outputs, _ = model.roi_heads(images, features, proposals, None)
        for i, instances in enumerate(outputs):
            feature = [features[key][i: i + 1]
                       for key in model.roi_heads.in_features]
            instances.roi_features = model.roi_heads.mask_pooler(  # TODO
                feature, [instances.pred_boxes])
        return outputs

    def postprocess(self, outputs, images, to_cpu):
        detections = []
        for instances, image in zip(outputs, images):
            height, width = image.shape[:2]
            instances = detector_postprocess(instances, height, width)
            keep = [
                self.model_meta.thing_classes[pred_class] in TYPE_MAPPING
                for pred_class in instances.pred_classes]
            instances = instances[keep]
            object_types = torch.as_tensor([
                TYPE_MAPPING[self.model_meta.thing_classes[pred_class]]
                for pred_class in instances.pred_classes],
                device=instances.pred_classes.device)
            features = instances.roi_features.mean(dim=(2, 3))
            features = features / features.norm(dim=1, keepdim=True)
            detection = Detection(
                instances.image_size, object_types=object_types,
                image_boxes=instances.pred_boxes,
                detection_scores=instances.scores,
                image_masks=instances.pred_masks, image_features=features)
            if self.interclass_nms_threshold is not None and len(detection) > 0:
                keep = nms(detection.image_boxes, detection.detection_scores,
                           self.interclass_nms_threshold)
                detection = detection[keep]
            if to_cpu:
                detection = detection.to('cpu')
            detections.append(detection)
        return detections

    def __call__(self, images, to_cpu=True):
        with torch.no_grad():
            images_processed = self.preprocess(images)
            outputs = self.inference(images_processed)
            detections = self.postprocess(outputs, images, to_cpu)
        return detections
