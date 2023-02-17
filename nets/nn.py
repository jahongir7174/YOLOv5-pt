import math

import torch

anchors = [[11,  12,  14,  32,  35,  24],
           [32,  61,  72,  56,  62,  141],
           [138, 109, 165, 243, 380, 334]]


def pad(k, p):
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


def check_anchor_order(m):
    a = m.anchors.prod(-1).mean(-1).view(-1)
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        m.anchors[:] = m.anchors.flip(0)


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p), 1, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 1e-3, 0.03)
        self.relu = torch.nn.SiLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 1),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(in_ch=out_ch, out_ch=out_ch)
        self.res_m = torch.nn.Sequential(*[Residual(out_ch // 2, add) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.res_m(self.conv1(x)), self.conv2(x)), dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, filters, num_dep):
        super().__init__()
        p1 = [Conv(filters[0], filters[1], 6, 2, 2)]
        p2 = [Conv(filters[1], filters[2], 3, 2),
              CSP(filters[2], filters[2], num_dep[0])]
        p3 = [Conv(filters[2], filters[3], 3, 2),
              CSP(filters[3], filters[3], num_dep[1])]
        p4 = [Conv(filters[3], filters[4], 3, 2),
              CSP(filters[4], filters[4], num_dep[2])]
        p5 = [Conv(filters[4], filters[5], 3, 2),
              CSP(filters[5], filters[5], num_dep[0]),
              SPP(filters[5], filters[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, filters, num_dep):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = Conv(filters[5], filters[4], 1, 1)
        self.h2 = CSP(2 * filters[4], filters[4], num_dep[0], False)
        self.h3 = Conv(filters[4], filters[3], 1, 1)
        self.h4 = CSP(2 * filters[3], filters[3], num_dep[0], False)
        self.h5 = Conv(filters[3], filters[3], 3, 2)
        self.h6 = CSP(2 * filters[3], filters[4], num_dep[0], False)
        self.h7 = Conv(filters[4], filters[4], 3, 2)
        self.h8 = CSP(2 * filters[4], filters[5], num_dep[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(p5)
        h2 = self.h2(torch.cat([self.up(h1), p4], 1))

        h3 = self.h3(h2)
        h4 = self.h4(torch.cat([self.up(h3), p3], 1))

        h5 = self.h5(h4)
        h6 = self.h6(torch.cat([h5, h3], 1))

        h7 = self.h7(h6)
        h8 = self.h8(torch.cat([h7, h1], 1))
        return h4, h6, h8


class Head(torch.nn.Module):
    stride = None

    def __init__(self, nc, filters):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = torch.nn.ModuleList(torch.nn.Conv2d(x, self.no * self.na, 1) for x in filters)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self.make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else torch.cat(z, 1)

    def make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, num_class):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(num_class, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.head.anchors = self.head.anchors / self.head.stride.view(-1, 1, 1)
        check_anchor_order(self.head)
        self.initialize_biases()
        self.stride = self.head.stride

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def initialize_biases(self):
        for m, s in zip(self.head.m, self.head.stride):
            b = m.bias.view(self.head.na, -1).clone()
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (self.head.nc - 0.99))
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v5_n(num_class: int = 80):
    depth = [1, 2, 3]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_class)


def yolo_v5_s(num_class: int = 80):
    depth = [1, 2, 3]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_class)


def yolo_v5_m(num_class: int = 80):
    depth = [2, 4, 6]
    width = [3, 48, 96, 192, 384, 768]
    return YOLO(width, depth, num_class)


def yolo_v5_l(num_class: int = 80):
    depth = [3, 6, 9]
    width = [3, 64, 128, 256, 512, 1024]
    return YOLO(width, depth, num_class)


def yolo_v5_x(num_class: int = 80):
    depth = [4, 8, 12]
    width = [3, 80, 160, 320, 640, 1280]
    return YOLO(width, depth, num_class)
