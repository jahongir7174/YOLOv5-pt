import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data

from utils import util


def input_fn(file_names, args, stride, params=None, augment=False):
    if args.distributed and augment:
        with util.distributed_manager(args.local_rank):
            dataset = Dataset(file_names, args.image_size, params, augment, int(stride))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        dataset = Dataset(file_names, args.image_size, params, augment, int(stride))
        sampler = None
    num_worker = os.cpu_count() // args.world_size
    loader = data.DataLoader(dataset, args.batch_size, sampler=sampler,
                             num_workers=num_worker, pin_memory=True, collate_fn=Dataset.collate_fn)
    return loader, dataset


class Dataset(data.Dataset):
    def __init__(self, file_names, image_size, params, augment=False, stride=32):
        self.params = params
        self.stride = stride
        self.augment = augment
        self.image_size = image_size
        self.mosaic_border = [-image_size // 2, -image_size // 2]

        cache = self.cache_labels(file_names)

        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = numpy.array(shapes, dtype=numpy.float64)
        self.file_names = list(cache.keys())

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        params = self.params
        mosaic = self.augment and random.random() < params['mosaic']
        if mosaic:
            shapes = None
            image, labels = self.load_mosaic(index)

            if random.random() < params['mixup']:
                img2, labels2 = self.load_mosaic(random.randint(0, len(self.file_names) - 1))
                ratio = numpy.random.beta(8.0, 8.0)
                image = (image * ratio + img2 * (1 - ratio)).astype(numpy.uint8)
                labels = numpy.concatenate((labels, labels2), 0)

        else:
            image, (h0, w0), (h, w) = self.load_image(index)

            shape = self.image_size
            image, ratio, pad = resize(image, shape, scale_up=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xy_wh to pixel xy_xy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            if not mosaic:
                image, labels = random_perspective(image, labels,
                                                   degree_gain=params['degrees'],
                                                   translate_gain=params['translate'],
                                                   scale_gain=params['scale'],
                                                   shear_gain=params['shear'],
                                                   perspective_gain=params['perspective'])
            # Augment colorspace
            augment_hsv(image, h_gain=params['hsv_h'], s_gain=params['hsv_s'], v_gain=params['hsv_v'])

        num_labels = len(labels)  # number of labels
        if num_labels:
            labels[:, 1:5] = util.xy2wh(labels[:, 1:5])  # convert xy_xy to xy_wh
            labels[:, [2, 4]] /= image.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= image.shape[1]  # normalized width 0-1

        if self.augment:
            if random.random() < params['fliplr']:
                image = numpy.fliplr(image)
                if num_labels:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((num_labels, 6))
        if num_labels:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # BGR -> RGB, HxWxC -> CxHxW
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = numpy.ascontiguousarray(image)

        return torch.from_numpy(image), labels_out, self.file_names[index], shapes

    @staticmethod
    def cache_labels(file_names):
        x = {}
        for file_name in file_names:
            try:
                shape = Image.open(file_name).size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                with open(file_name.replace('images', 'labels').replace('jpg', 'txt'), 'r') as f:
                    y = numpy.array([x.split() for x in f.read().splitlines()], dtype=numpy.float32)
                if len(y) == 0:
                    y = numpy.zeros((0, 5), dtype=numpy.float32)
                x[file_name] = [y, shape]
            except FileNotFoundError:
                pass
        return x

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def load_image(self, index):
        path = self.file_names[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def load_mosaic(self, index):
        img4 = None
        labels4 = []
        s = self.image_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
        indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = None, None, None, None, None, None, None, None,
        for i, index in enumerate(indices):
            img, _, (h, w) = self.load_image(index)

            if i == 0:
                img4 = numpy.full((s * 2, s * 2, img.shape[2]), 114, dtype=numpy.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            # Labels
            x = self.labels[index]
            labels = x.copy()
            if x.size > 0:
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + pad_w
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + pad_h
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + pad_w
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + pad_h
            labels4.append(labels)

        if len(labels4):
            labels4 = numpy.concatenate(labels4, 0)
            numpy.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

        # Augment
        img4, labels4 = random_perspective(img4, labels4,
                                           degree_gain=self.params['degrees'],
                                           translate_gain=self.params['translate'],
                                           scale_gain=self.params['scale'],
                                           shear_gain=self.params['shear'],
                                           perspective_gain=self.params['perspective'],
                                           border=self.mosaic_border)

        return img4, labels4


def augment_hsv(image, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    r = numpy.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=numpy.int16)
    lut_hue = ((x * r[0]) % 180).astype('uint8')
    lut_sat = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_val = numpy.clip(x * r[2], 0, 255).astype('uint8')

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype('uint8')
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def resize(image, new_shape=(640, 640), color=(114, 114, 114), scale_up=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_pad[0], new_shape[0] - new_pad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_pad:  # resize
        image = cv2.resize(image, new_pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return image, ratio, (dw, dh)


def random_perspective(image, targets=(), degree_gain=10, translate_gain=.1,
                       scale_gain=.1, shear_gain=10, perspective_gain=0.0, border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2

    # Center
    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)
    perspective[2, 0] = random.uniform(-perspective_gain, perspective_gain)  # x perspective (about y)
    perspective[2, 1] = random.uniform(-perspective_gain, perspective_gain)  # y perspective (about x)

    # Rotation and Scale
    rotation = numpy.eye(3)
    a = random.uniform(-degree_gain, degree_gain)
    s = random.uniform(1 - scale_gain, 1 + scale_gain)
    rotation[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-shear_gain, shear_gain) * math.pi / 180)  # x shear (deg)
    shear[1, 0] = math.tan(random.uniform(-shear_gain, shear_gain) * math.pi / 180)  # y shear (deg)

    # Translation
    translation = numpy.eye(3)
    translation[0, 2] = random.uniform(0.5 - translate_gain, 0.5 + translate_gain) * w  # x translation (pixels)
    translation[1, 2] = random.uniform(0.5 - translate_gain, 0.5 + translate_gain) * h  # y translation (pixels)

    # Combined rotation matrix
    matrix = translation @ shear @ rotation @ perspective @ center  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():  # image changed
        if perspective_gain:
            image = cv2.warpPerspective(image, matrix, dsize=(w, h), borderValue=(114, 114, 114))
        else:  # affine
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        if perspective_gain:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return image, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates
