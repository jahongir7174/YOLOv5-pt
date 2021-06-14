import contextlib
import copy
import math
import random
import time

import numpy
import torch
import torchvision
import tqdm
from scipy.cluster.vq import kmeans
from torch import distributed


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_anchors(dataset, model, thr=4.0, img_size=640):
    m = model.module.head if hasattr(model, 'module') else model.head
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = numpy.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(numpy.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        return (best > 1. / thr).float().mean(), (x > 1. / thr).float().sum(1).mean()

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print(f'anchors/target = {aat:.2f}, best possible recall (BPR) = {bpr:.4f}', end='')
    if bpr < 0.98:  # threshold to recompute
        print('\nAnalyzing anchors... ', end='')
        print('anchors/target = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr), end='')
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2  # number of anchors
        new_anchors = k_means_anchors(dataset, n=na, img_size=img_size, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = new_anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print('New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print('Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def k_means_anchors(dataset, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    thr = 1. / thr

    def metric(k_, wh_):  # compute metrics
        r = wh_[:, None] / k_[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k_):  # mutation fitness
        _, best = metric(torch.tensor(k_, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k_):
        k_ = k_[numpy.argsort(k_.prod(1))]  # sort small to large
        x, best = metric(k_, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anchor > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i_, x in enumerate(k_):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i_ < len(k_) - 1 else '\n')  # use in *.cfg
        return k_

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = numpy.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # K-means calculation
    print('Running k-means for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    npr = numpy.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    p_bar = tqdm.tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in p_bar:
        v = numpy.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            p_bar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def ap_per_class(tp, conf, pred_cls, target_cls):
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = numpy.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = numpy.zeros(s), numpy.zeros(s), numpy.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            # r at pr_score, negative x, xp because xp decreases
            r[ci] = numpy.interp(-pr_score, -conf[i], recall[:, 0])

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = numpy.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], m_pre, m_rec = compute_ap(recall[:, j], precision[:, j])
                if j == 0:
                    py.append(numpy.interp(px, m_rec, m_pre))  # precision at mAP@0.5

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    # Append sentinel values to beginning and end
    m_rec = recall  # np.concatenate(([0.], recall, [recall[-1] + 1E-3]))
    m_pre = precision  # np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate
    else:  # 'continuous'
        i = numpy.where(m_rec[1:] != m_rec[:-1])[0]  # points where x axis (recall) changes
        ap = numpy.sum((m_rec[i + 1] - m_rec[i]) * m_pre[i + 1])  # area under curve

    return ap, m_pre, m_rec


def fitness(x):
    w = [0.1, 0.0, 0.1, 0.8]
    return (x[:, :4] * w).sum(1)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def model_info(model, img_size):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)

    try:
        from thop import profile
        stride = int(model.head.stride.max())
        flops = profile(copy.deepcopy(model), inputs=(torch.zeros(1, 3, stride, stride),), verbose=False)[0] / 1E9 * 2
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)
    except (ImportError, Exception):
        fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def init_seeds(seed=0):
    import torch.backends.cudnn
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = numpy.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(numpy.int)  # labels = [class xy-wh]
    weights = numpy.bincount(classes, minlength=nc)  # occurrences per class

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def xy2wh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def wh2xy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coordinates(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xy_xy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coordinates(coords, img0_shape)
    return coords


def clip_coordinates(boxes, img_shape):
    # Clip bounding xy_xy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_threshold=0.1, iou_threshold=0.6, merge=False, classes=None, agnostic=False):
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_threshold  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(x[:, :4])

        # Detections matrix nx6 (xy_xy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_threshold  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def strip_optimizer(file_name):
    x = torch.load(file_name, map_location=torch.device('cpu'))
    x['model'].half()
    for param in x['model'].parameters():
        param.requires_grad = False
    torch.save(x, file_name)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


@contextlib.contextmanager
def distributed_manager(local_rank: int):
    if local_rank != 0:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def is_parallel(model):
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


class EMA:
    def __init__(self, model):
        self.num = 0
        self.ema = copy.deepcopy(model.module if is_parallel(model) else model).eval()

        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.num += 1
            decay = 0.9999 * (1 - math.exp(-self.num / 2000))

            m_std = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1. - decay) * m_std[k].detach()


def smooth_bce(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_iou(box1, box2, x1y1x2y2=True, g_iou=False, d_iou=False, c_iou=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xy_wh to xy_xy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if g_iou or d_iou or c_iou:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if c_iou or d_iou:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if d_iou:
                return iou - rho2 / c2  # DIoU
            elif c_iou:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class FocalLoss(torch.nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, params):
        super().__init__()
        device = next(model.parameters()).device  # get model device

        # Define criteria
        bce_cls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([params['cls_pw']], device=device))
        bce_obj = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([params['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_bce(eps=0.0)  # positive, negative BCE targets

        # Focal loss
        if params['fl_gamma'] > 0:
            bce_cls, bce_obj = FocalLoss(bce_cls, params['fl_gamma']), FocalLoss(bce_obj, params['fl_gamma'])

        det = model.module.head if is_parallel(model) else model.head
        self.balance = [8.0, 4.0, 1.0, 0.4]
        self.bce_cls, self.bce_obj, self.hyp = bce_cls, bce_obj, params

        self.na = det.num_anchor
        self.nc = det.num_class
        self.nl = det.num_layers
        self.anchors = det.anchors

    def __call__(self, y_pred, y_true):
        na, nt = self.na, y_true.shape[0]  # number of anchors, targets
        t_cls, t_box, indices, anchors = [], [], [], []
        gain = torch.ones(7, device=y_true.device)  # normalized to grid-space gain
        ai = torch.arange(na, device=y_true.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        true = torch.cat((y_true.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], ], device=true.device).float() * g  # offsets

        for i in range(self.nl):
            anchor = self.anchors[i]
            gain[2:6] = torch.tensor(y_pred[i].shape)[[3, 2, 3, 2]]  # xy-xy gain

            # Match targets to anchors
            t = true * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchor[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = true[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            t_box.append(torch.cat((gxy - gij, gwh), 1))  # box
            anchors.append(anchor[a])  # anchors
            t_cls.append(c)  # class

        device = y_true.device
        l_cls = torch.zeros(1, device=device)
        l_box = torch.zeros(1, device=device)
        l_obj = torch.zeros(1, device=device)
        t_obj = None
        # Losses
        for i, pred in enumerate(y_pred):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, grid_y, grid_x
            t_obj = torch.zeros_like(pred[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pred[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                p_xy = ps[:, :2].sigmoid() * 2. - 0.5
                p_wh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                p_box = torch.cat((p_xy, p_wh), 1)  # predicted box
                iou = compute_iou(p_box.T, t_box[i], c_iou=True)  # iou(prediction, target)
                l_box += (1.0 - iou).mean()  # iou loss

                # Object-ness
                t_obj[b, a, gj, gi] = iou.detach().clamp(0).type(t_obj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), t_cls[i]] = self.cp
                    l_cls += self.bce_cls(ps[:, 5:], t)  # BCE

            l_obj += self.bce_obj(pred[..., 4], t_obj) * self.balance[i]  # obj loss

        l_box *= self.hyp['box']
        l_obj *= self.hyp['obj']
        l_cls *= self.hyp['cls']
        bs = t_obj.shape[0]  # batch size

        loss = l_box + l_obj + l_cls
        return loss * bs, loss.detach()
