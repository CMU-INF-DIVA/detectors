import os
import os.path as osp
import sys

import cv2
import torch
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

sys.path.insert(0, '../..')

import detectors
from detectors.visualizer import Visualizer


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


def draw_grid(name, backbones, input_shapes, images, set_shape):
    visualizer = Visualizer()
    n_row = len(backbones) * len(images)
    n_col = len(input_shapes)
    fig, axes = plt.subplots(n_row, n_col, figsize=(
        n_col * 16 / 3, n_row * 9 / 3), dpi=120)
    fig.suptitle(name)
    fig.set_tight_layout(0.01)
    bar = tqdm(total=n_row * n_col)
    for i1, backbone in enumerate(backbones):
        model = detectors.get(name)(0, backbone, input_shapes[0])
        for i2, input_shape in enumerate(input_shapes):
            set_shape(model, input_shape)
            for i3, img in enumerate(images.values()):
                detection = model([img])[0]
                visual_image = visualizer.draw(img, detection, show=False)
                ax = axes[i1 * len(images) + i3, i2]
                ax.imshow(visual_image)
                ax.set_title('%s %s' % (backbone, input_shape))
                ax.axis('off')
                bar.update()
    bar.close()
