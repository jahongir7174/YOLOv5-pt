import math

import torch

anchor = [[10,  11, 24,  24,  27,  52],
          [27,  20, 25,  45,  53,  41],
          [45,  89, 97,  79,  79,  170],
          [234, 99, 171, 195, 327, 272]]


def check_anchor_order(m):
    a = m.anchor_grid.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, k // 2, 1, g, False)
        self.norm = torch.nn.GroupNorm(40, out_ch, 1e-3)
        self.silu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.norm(self.conv(x)))


class Residual(torch.nn.Module):
    def __init__(self, ch, add):
        super().__init__()
        self.add = add
        self.res = torch.nn.Sequential(Conv(ch, ch, 1),
                                       Conv(ch, ch, 3))

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(out_ch, out_ch)
        self.res_m = torch.nn.Sequential(*[Residual(out_ch // 2, add) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.conv1(x), self.res_m(self.conv2(x))), dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=(5, 9, 13)):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, 1, 1)
        self.conv2 = Conv(in_ch, out_ch, 1, 1)
        self.conv3 = Conv(out_ch, out_ch, 3, 1)
        self.conv4 = Conv(out_ch, out_ch, 1, 1)
        self.conv5 = Conv(4 * out_ch, out_ch, 1, 1)
        self.conv6 = Conv(out_ch, out_ch, 3, 1)
        self.conv7 = Conv(2 * out_ch, out_ch, 1, 1)
        self.res_m = torch.nn.ModuleList([torch.nn.MaxPool2d(x, 1, x // 2) for x in k])

    def forward(self, x):
        x1 = x
        y1 = self.conv1(x1)
        x2 = self.conv4(self.conv3(self.conv2(x)))
        y2 = self.conv6(self.conv5(torch.cat([x2] + [m(x2) for m in self.res_m], 1)))
        return self.conv7(torch.cat((y1, y2), dim=1))


class Head(torch.nn.Module):
    stride = None
    export = False

    def __init__(self, num_class, anchors, filters):
        super().__init__()
        self.num_class = num_class
        self.num_output = num_class + 5
        self.num_layers = len(anchors)
        self.num_anchor = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.num_layers
        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.num_layers, 1, -1, 1, 1, 2))
        self.m = torch.nn.ModuleList(torch.nn.Conv2d(x, self.num_output * self.num_anchor, 1) for x in filters)

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.num_layers):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.num_anchor, self.num_output, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self.make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.num_output))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def make_grid(nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class DarkNet(torch.nn.Module):
    def __init__(self, filters, num_dep, gate):
        super().__init__()
        b1 = [Conv(filters[0], filters[1], 3, 1)]
        b2 = [Conv(filters[1], filters[2], 3, 2),
              CSP(filters[2], filters[2], num_dep[0], gate[0])]
        b3 = [Conv(filters[2], filters[3], 3, 2),
              CSP(filters[3], filters[3], num_dep[1], gate[0])]
        b4 = [Conv(filters[3], filters[4], 3, 2),
              CSP(filters[4], filters[4], num_dep[1], gate[0])]
        b5 = [Conv(filters[4], filters[5], 3, 2),
              SPP(filters[5], filters[5]),
              CSP(filters[5], filters[5], num_dep[0], gate[1])]

        self.b1 = torch.nn.Sequential(*b1)
        self.b2 = torch.nn.Sequential(*b2)
        self.b3 = torch.nn.Sequential(*b3)
        self.b4 = torch.nn.Sequential(*b4)
        self.b5 = torch.nn.Sequential(*b5)

    def forward(self, x):
        b1 = self.b1(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        return b5, b4, b3, b2, b1


class YOLO(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()
        num_dep = [4, 12]
        filters = [12, 80, 160, 320, 640, 1280]

        self.up = torch.nn.Upsample(None, 2)

        self.net = DarkNet(filters, num_dep, [True, False])

        self.h10 = Conv(filters[5], filters[4], 3, 1)
        self.h11 = CSP(filters[5], filters[4], num_dep[0], False)
        self.h12 = Conv(filters[4], filters[3], 3, 1)
        self.h13 = CSP(filters[4], filters[3], num_dep[0], False)
        self.h14 = Conv(filters[3], filters[2], 3, 1)
        self.h15 = CSP(filters[3], filters[2], num_dep[0], False)
        self.h16 = Conv(filters[2], filters[2], 3, 2)
        self.h17 = CSP(filters[3], filters[3], num_dep[0], False)
        self.h18 = Conv(filters[3], filters[3], 3, 2)
        self.h19 = CSP(filters[4], filters[4], num_dep[0], False)
        self.h20 = Conv(filters[4], filters[4], 3, 2)
        self.h21 = CSP(filters[5], filters[5], num_dep[0], False)

        self.head = Head(num_class, anchor, (filters[2], filters[3], filters[4], filters[5]))

        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(torch.zeros(1, 3, 256, 256))])
        self.head.anchors /= self.head.stride.view(-1, 1, 1)
        check_anchor_order(self.head)
        self.initialize_biases()

    def forward(self, x):
        b5, b4, b3, b2, b1 = self.net(x)

        h10 = self.h10(b5)
        h11 = self.h11(torch.cat([self.up(h10), b4], 1))

        h12 = self.h12(h11)
        h13 = self.h13(torch.cat([self.up(h12), b3], 1))

        h14 = self.h14(h13)
        h15 = self.h15(torch.cat([self.up(h14), b2], 1))

        h16 = self.h16(h15)
        h17 = self.h17(torch.cat([h16, h14], 1))

        h18 = self.h18(h17)
        h19 = self.h19(torch.cat([h18, h12], 1))

        h20 = self.h20(h19)
        h21 = self.h21(torch.cat([h20, h10], 1))
        return self.head([h15, h17, h19, h21])

    def initialize_biases(self):
        for mi, s in zip(self.head.m, self.head.stride):
            b = mi.bias.view(self.head.num_anchor, -1).clone()
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (self.head.num_class - 0.99))
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


class Ensemble(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        y = []
        for module in self.modules:
            y.append(module(x, True)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y
