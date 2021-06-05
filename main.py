import argparse
import copy
import math
import os
import random

import numpy
import torch
import tqdm
import yaml

from nets import nn
from utils import util
from utils.dataset import input_fn


def learning_rate(params, epochs):
    def fn(x):
        return ((1 - math.cos(x * math.pi / epochs)) / 2) * (params['lrf'] - 1) + 1
    return fn


def train(params, args, device):
    epochs = 300
    util.init_seeds()

    model = nn.YOLO(len(params['names'])).to(device)
    if os.path.exists('weights/coco_best.pt'):
        checkpoint = torch.load('weights/coco_best.pt', device)
        state_dict = checkpoint['model'].float().state_dict()
        state_dict = util.intersect_dicts(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=False)

    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, torch.nn.GroupNorm):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            pg1.append(v.weight)

    optimizer = torch.optim.SGD(pg0, lr=params['lr0'], momentum=params['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    del pg0, pg1, pg2

    lr = learning_rate(params, epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    stride = max(int(model.head.stride.max()), 32)  # grid size (max stride)
    num_layers = model.head.num_layers

    # DP mode
    if not args.distributed and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    file_names = []
    with open(os.path.join('../Dataset/COCO/train2017.txt')) as f:
        for file_name in f.readlines():
            file_name = os.path.basename(file_name.rstrip())
            file_names.append(f'../Dataset/COCO/images/train2017/{file_name}')
    loader, dataset = input_fn(file_names, args, stride, params, True)

    # Process 0
    if args.local_rank == 0:
        util.check_anchors(dataset, model, params['anchor_t'], args.image_size)
        model.half().float()

    # DDP mode
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    params['box'] *= 3. / num_layers
    params['cls'] *= len(params['names']) / 80. * 3. / num_layers
    params['obj'] *= (args.image_size / 640) ** 2 * 3. / num_layers
    model.class_weights = util.labels_to_class_weights(dataset.labels, len(params['names'])).to(device)
    model.names = params['names']

    num_warmup = max(round(params['warmup_epochs'] * len(loader)), 1000)
    scheduler.last_epoch = -1
    amp_scale = torch.cuda.amp.GradScaler()
    compute_loss = util.ComputeLoss(model, params)
    best_fitness = 0.0
    for epoch in range(0, epochs):
        model.train()

        m_loss = torch.zeros(1, device=device)
        if args.distributed:
            loader.sampler.set_epoch(epoch)
        p_bar = enumerate(loader)
        if args.local_rank == 0:
            print(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem', 'loss'))
            p_bar = tqdm.tqdm(p_bar, total=len(loader))
        optimizer.zero_grad()
        for i, (images, target, _, _) in p_bar:
            ni = i + len(loader) * epoch
            images = images.to(device, non_blocking=True).float() / 255.0

            # Warmup
            if ni <= num_warmup:
                xi = [0, num_warmup]
                accumulate = max(1, numpy.interp(ni, xi, [1, 64 / (args.batch_size * args.world_size)]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = numpy.interp(ni, xi,
                                           [params['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lr(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = numpy.interp(ni, xi, [params['warmup_momentum'], params['momentum']])

            # Multi-scale
            if args.multi_scale:
                sz = random.randrange(args.imag_size * 0.5, args.imeage_size * 1.5 + stride) // stride * stride  # size
                sf = sz / max(images.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / stride) * stride for x in
                          images.shape[2:]]  # new shape (stretched to gs-multiple)
                    images = torch.nn.functional.interpolate(images, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast():
                loss, loss_items = compute_loss(model(images), target.to(device))  # loss scaled by batch_size
                if args.distributed:
                    loss *= args.world_size  # gradient averaged between devices in DDP mode

            # Backward
            amp_scale.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                amp_scale.step(optimizer)  # optimizer.step
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            if args.local_rank == 0:
                m_loss = (m_loss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ('%10s' * 2 + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), mem, *m_loss)
                p_bar.set_description(s)

        # Scheduler
        scheduler.step()

        # DDP process 0 or single-GPU
        if args.local_rank == 0:

            current = util.fitness(numpy.array(test(ema.ema, args, params)).reshape(1, -1))
            if current > best_fitness:
                best_fitness = current

            save = {'model': copy.deepcopy(ema.ema).half()}

            torch.save(save, 'weights/coco_last.pt')
            if best_fitness == current:
                torch.save(save, 'weights/coco_best.pt')
            del save

    if args.local_rank == 0:
        util.strip_optimizer('weights/coco_last.pt')
        util.strip_optimizer('weights/coco_best.pt')
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def test(model=None, args=None, params=None):
    if model is not None:
        device = next(model.parameters()).device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(os.path.join('weights', 'best.pt'), device)['model'].float().eval()

    half = device.type != 'cpu'
    if half:
        model.half()

    model.eval()

    iou_v = torch.linspace(0.5, 0.95, 10).to(device)
    n_iou = iou_v.numel()

    stride = max(int(model.head.stride.max()), 32)
    file_names = []
    with open(os.path.join('../Dataset/COCO/val2017.txt')) as f:
        for file_name in f.readlines():
            file_name = os.path.basename(file_name.rstrip())
            file_names.append(f'../Dataset/COCO/images/val2017/{file_name}')
    loader = input_fn(file_names, args, stride, params)[0]
    seen = 0
    s = ('%10s' * 3) % ('precision', 'recall', 'mAP')
    p, r, f1, mp, mr, map50, mean_ap, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    for images, target, paths, shapes in tqdm.tqdm(loader, desc=s):
        images = images.to(device, non_blocking=True)
        images = images.half() if half else images.float()
        images /= 255.0
        target = target.to(device)
        _, _, height, width = images.shape
        wh_wh = torch.Tensor([width, height, width, height]).to(device)

        with torch.no_grad():
            t = util.time_synchronized()
            inf_out, train_out = model(images)
            t0 += util.time_synchronized() - t
            t = util.time_synchronized()
            output = util.non_max_suppression(inf_out, 0.001)
            t1 += util.time_synchronized() - t

        for si, pred in enumerate(output):
            labels = target[target[:, 0] == si, 1:]
            nl = len(labels)
            t_cls = labels[:, 0].tolist() if nl else []
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, n_iou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), t_cls))
                continue

            pred_n = pred.clone()
            util.scale_coordinates(images[si].shape[1:], pred_n[:, :4], shapes[si][0], shapes[si][1])
            correct = torch.zeros(pred.shape[0], n_iou, dtype=torch.bool, device=device)
            if nl:
                detected = []
                t_cls_tensor = labels[:, 0]
                t_box = util.wh2xy(labels[:, 1:5]) * wh_wh
                util.scale_coordinates(images[si].shape[1:], t_box, shapes[si][0], shapes[si][1])

                for cls in torch.unique(t_cls_tensor):
                    ti = (cls == t_cls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    if pi.shape[0]:
                        iou_list, i = util.box_iou(pred_n[pi, :4], t_box[ti]).max(1)

                        detected_set = set()
                        for j in (iou_list > iou_v[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = iou_list[j] > iou_v
                                if len(detected) == nl:
                                    break

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), t_cls))

    stats = [numpy.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = util.ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)
        mp, mr, map50, mean_ap = p.mean(), r.mean(), ap50.mean(), ap.mean()

    print('%10.3g' * 3 % (mp, mr, mean_ap))

    if model is None:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1))
        s = f'Speed: {t[0]:.1f}/{t[1]:.1f}/{t[2]:.1f} ms inference/nms/total'
        print(f'{s} per {args.image_size}x{args.image_size} image at batch-size {args.batch_size}')

    model.float()
    return mp, mr, map50, mean_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=416)
    parser.add_argument('--batch-size', type=int, default=34)
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda:0')

    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = True
    else:
        args.world_size = 1
        args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    with open(os.path.join('utils', 'args.yaml')) as f:
        params = yaml.safe_load(f)

    train(params, args, device)


if __name__ == '__main__':
    main()
