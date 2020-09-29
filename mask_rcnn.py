import warnings

import torch
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import get_checkpoint_url, get_config_file
from detectron2.modeling import build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import ImageList
from torchvision.ops.boxes import nms

from .base import Detection, Detector, ObjectType
from .utils import FvcoreCachePath


class MaskRCNN(Detector):

    CFG_FILES = {
        'res50': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'res101': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        'res101x': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    }

    TYPE_MAPPING = {
        'person': ObjectType.Person, 'car': ObjectType.Vehicle,
        'bus': ObjectType.Vehicle, 'truck': ObjectType.Vehicle,
        'motorcycle': ObjectType.Vehicle, 'bicycle': ObjectType.Bike
    }

    def __init__(self, gpu_id=None,
                 model='res101',
                 input_shape=(1920, 1080),
                 output_mask=False,
                 output_feature=False,
                 score_threshold=0.5,
                 interclass_nms_threshold=None):
        assert model in self.CFG_FILES, 'Unsupported model %s' % (model)
        super(MaskRCNN, self).__init__(gpu_id)
        cfg_file = self.CFG_FILES[model]
        cfg = get_cfg()
        cfg.merge_from_file(get_config_file(cfg_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.MASK_ON = output_mask
        model = build_model(cfg)
        with FvcoreCachePath():
            DetectionCheckpointer(model).load(get_checkpoint_url(cfg_file))
        model.requires_grad_(False)
        model.eval()
        self.model = model
        self.cfg = cfg
        self.model_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        input_shape = sorted(input_shape, reverse=True)
        self.input_shape = input_shape + [input_shape[0] / input_shape[1]]
        self.output_mask = output_mask
        self.output_feature = output_feature
        self.interclass_nms_threshold = interclass_nms_threshold

    def preprocess(self, images):
        processed_images = []
        for image in images:
            height, width = image.shape[:2]
            image = image.to(device=self.device, non_blocking=True)
            image = image.permute(2, 0, 1).type(torch.float)
            origin_ratio = width / height
            max_size, min_size, ratio = self.input_shape
            if ratio > origin_ratio:
                target_height = min_size
                target_width = int(round(target_height * origin_ratio))
            else:
                target_width = max_size
                target_height = int(round(target_width / origin_ratio))
            target_shape = (target_height, target_width)
            image = F.interpolate(
                image.unsqueeze(0), target_shape,
                mode='bilinear', align_corners=False)
            image = (image.squeeze(0) - self.model.pixel_mean) / \
                self.model.pixel_std
            processed_images.append(image)
        processed_images = ImageList.from_tensors(
            processed_images, self.model.backbone.size_divisibility)
        return processed_images

    def inference(self, images):
        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, None)
        outputs, _ = self.model.roi_heads(images, features, proposals, None)
        if self.output_feature:
            for i, instances in enumerate(outputs):
                feature = [features[key][i: i + 1]
                           for key in self.model.roi_heads.in_features]
                instances.roi_features = self.model.roi_heads.box_pooler(
                    feature, [instances.pred_boxes])
        return outputs

    def postprocess(self, outputs, images, to_cpu):
        detections = []
        for instances, image in zip(outputs, images):
            height, width = image.shape[:2]
            instances = detector_postprocess(instances, height, width)
            keep = [
                self.model_meta.thing_classes[pred_class] in self.TYPE_MAPPING
                for pred_class in instances.pred_classes]
            instances = instances[keep]
            object_types = torch.as_tensor([
                self.TYPE_MAPPING[self.model_meta.thing_classes[pred_class]]
                for pred_class in instances.pred_classes],
                device=instances.pred_classes.device)
            detection = Detection(
                instances.image_size, object_types=object_types,
                image_boxes=instances.pred_boxes.tensor,
                detection_scores=instances.scores)
            if self.output_mask:
                detection.image_masks = instances.pred_masks
            if self.output_feature:
                features = instances.roi_features.mean(dim=(2, 3))
                features = features / features.norm(dim=1, keepdim=True)
                detection.image_features = features
            if self.interclass_nms_threshold is not None and len(detection) > 0:
                keep = nms(detection.image_boxes, detection.detection_scores,
                           self.interclass_nms_threshold)
                detection = detection[keep]
            if to_cpu:
                detection = detection.to('cpu')
            detections.append(detection)
        return detections

    def __call__(self, images, to_cpu=True):
        with torch.no_grad() and warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            processed_images = self.preprocess(images)
            outputs = self.inference(processed_images)
            detections = self.postprocess(outputs, images, to_cpu)
        return detections
