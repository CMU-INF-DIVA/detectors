import os
import os.path as osp
import sys
from itertools import product

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

sys.path.insert(0, '../..')

import detectors  # noqa
from detectors.visualizer import Visualizer  # noqa


def load_images(dirname=osp.dirname(__file__)):
    images = {}
    for filename in os.listdir(dirname):
        if not filename.endswith('.png'):
            continue
        path = osp.join(dirname, filename)
        image = cv2.imread(path)
        image = torch.as_tensor(image, dtype=torch.float)
        images[filename] = image
    return images


def draw_once(model, images):
    visualizer = Visualizer()
    detections = []
    for image in images:
        detection = model([image])[0]
        visualizer.draw(image, detection)
        detections.append(detection)
    return detections


def draw_grid(name, params, images, n_col=4):
    visualizer = Visualizer()
    all_params = []
    for values in product(*params.values()):
        param = {k: v for k, v in zip(params.keys(), values)}
        all_params.append(param)
    n_images = len(all_params) * len(images)
    n_row = int(np.ceil(n_images / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(
        n_col * 16 / 3, n_row * 9 / 3), dpi=120)
    axes = axes.flatten()
    fig.suptitle(name)
    fig.set_tight_layout(0.01)
    ax_i = 0
    for param in tqdm(all_params):
        model = detectors.get(name)(0, **param)
        for image_name, image in images.items():
            detection = model([image])[0]
            visual_image = visualizer.draw(image, detection, show=False)
            ax = axes[ax_i]
            ax_i += 1
            ax.imshow(visual_image)
            ax.set_title('%s: %s' % (', '.join(
                [str(v) for v in param.values()]), image_name))
            ax.axis('off')
        del model
