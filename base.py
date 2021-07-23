from enum import IntEnum, auto
from typing import List, Union

import torch
from torch.backends import cudnn
from detectron2.structures import Instances

Detection = Instances
'''
Detection attributes: object_types, image_boxes, detection_scores, \
    [image_features, image_masks]
'''


class ObjectType(IntEnum):

    Vehicle = auto()
    Person = auto()
    Bike = auto()
    TrafficLight = auto()


class COCOObjectType(IntEnum):

    Person = 0
    Bicycle = auto()
    Car = auto()
    Motorcycle = auto()
    Airplane = auto()
    Bus = auto()
    Train = auto()
    Truck = auto()
    Boat = auto()
    Traffic_light = auto()
    Fire_hydrant = auto()
    Stop_sign = auto()
    Parking_meter = auto()
    Bench = auto()
    Bird = auto()
    Cat = auto()
    Dog = auto()
    Horse = auto()
    Sheep = auto()
    Cow = auto()
    Elephant = auto()
    Bear = auto()
    Zebra = auto()
    Giraffe = auto()
    Backpack = auto()
    Umbrella = auto()
    Handbag = auto()
    Tie = auto()
    Suitcase = auto()
    Frisbee = auto()
    Skis = auto()
    Snowboard = auto()
    Sports_ball = auto()
    Kite = auto()
    Baseball_bat = auto()
    Baseball_glove = auto()
    Skateboard = auto()
    Surfboard = auto()
    Tennis_racket = auto()
    Bottle = auto()
    Wine_glass = auto()
    Cup = auto()
    Fork = auto()
    Knife = auto()
    Spoon = auto()
    Bowl = auto()
    Banana = auto()
    Apple = auto()
    Sandwich = auto()
    Orange = auto()
    Broccoli = auto()
    Carrot = auto()
    Hot_dog = auto()
    Pizza = auto()
    Donut = auto()
    Cake = auto()
    Chair = auto()
    Couch = auto()
    Potted_plant = auto()
    Bed = auto()
    Dining_table = auto()
    Toilet = auto()
    Tv = auto()
    Laptop = auto()
    Mouse = auto()
    Remote = auto()
    Keyboard = auto()
    Cell_phone = auto()
    Microwave = auto()
    Oven = auto()
    Toaster = auto()
    Sink = auto()
    Refrigerator = auto()
    Book = auto()
    Clock = auto()
    Vase = auto()
    Scissors = auto()
    Teddy_bear = auto()
    Hair_drier = auto()
    Toothbrush = auto()


class Detector(object):

    def __init__(self, gpu_id: Union[None, int] = None):
        self.device = 'cpu'
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = 'cuda:%d' % (gpu_id)
            cudnn.fastest = True
            cudnn.benchmark = True

    def __call__(self, images: List[torch.Tensor],
                 to_cpu: bool = True) -> List[Detection]:
        '''
        images: a list of pytorch tensors as H x W x C[BGR] in [0, 256) on cpu
        to_cpu: whether to move detections to cpu
        '''
        raise NotImplementedError

    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.device)
