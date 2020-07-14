# Detectors

Author: Lijun Yu

Email: lijun@lj-y.com

A submodule of object detection models.

## Models

* Mask R-CNN from [Detectron2](https://github.com/facebookresearch/detectron2)
* YOLOv5 [official](https://github.com/ultralytics/yolov5)
* EfficientDet in [PyTorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

## API

```python
import torch
from detectors import get_detector
detector_class = get_detector('Mask R-CNN')  # Or YOLOv5, EfficientDet
detector = detector_class(gpu_id=0)
# images: a list of pytorch tensors as H x W x C[BGR] in [0, 256)
images = [torch.zeros(1080, 1920, 3)]
detections = detector(images)
```

## Dependency

See [actev_base](https://github.com/CMU-INF-DIVA/actev_base).

## License

See [License](LICENSE).
