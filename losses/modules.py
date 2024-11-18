import torch
import torch.nn as nn
import torch.nn.functional as F


def get_smoothness_func(smoothness_type):
    if smoothness_type == '1st_order':
        return depth_smoothness
    elif smoothness_type == '2nd_order':
        return depth_smoothness_2nd_order
    else:
        raise NotImplementedError


def get_reconstr_loss_func(reconstr_loss_type):
    if reconstr_loss_type == 'smooth_l1':
        return lambda warped, ref, mask: compute_reconstr_loss(warped, ref, mask, simple=False)
    elif reconstr_loss_type == 'smooth_l1_fixed':
        return compute_reconstr_loss_fixed
    else:
        raise NotImplementedError


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        # self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        # print('mask: {}'.format(mask.shape))
        # print('x: {}'.format(x.shape))
        # print('y: {}'.format(y.shape))
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        y = y.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        # x = self.refl(x)
        # y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask)
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def gradient(pred, append_zeros=False):
    D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    if append_zeros:
        zeros_row = torch.zeros_like(D_dy[:, :1])
        D_dy = torch.cat((D_dy, zeros_row), dim=1)
        zeros_col = torch.zeros_like(D_dx[:, :, :1])
        D_dx = torch.cat((D_dx, zeros_col), dim=2)
    return D_dx, D_dy


def depth_smoothness(depth, img, lambda_wt=1, **kwargs):
    """Computes image-aware depth smoothness loss."""
    # print('depth: {} img: {}'.format(depth.shape, img.shape))
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 3, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dy), 3, keepdim=True)))
    # print('depth_dx: {} weights_x: {}'.format(depth_dx.shape, weights_x.shape))
    # print('depth_dy: {} weights_y: {}'.format(depth_dy.shape, weights_y.shape))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))


def depth_smoothness_2nd_order(depth, img, lambda_wt=1.0, clip=2.0):
    """Computes image-aware depth smoothness loss using relaxed 2nd-order formulation."""
    depth_dx, depth_dy = gradient(depth, append_zeros=True)
    depth_dxdx, depth_dxdy = gradient(depth_dx, append_zeros=True)
    depth_dydx, depth_dydy = gradient(depth_dy, append_zeros=True)

    # enormous 2nd order gradients correspond to depth discontinuities
    depth_dxdx = torch.clip(depth_dxdx, -clip, clip)
    depth_dxdy = torch.clip(depth_dxdy, -clip, clip)
    depth_dydx = torch.clip(depth_dydx, -clip, clip)
    depth_dydy = torch.clip(depth_dydy, -clip, clip)

    image_dx, image_dy = gradient(img, append_zeros=True)

    # get weight for gradient penalty
    weights_x = torch.exp(-(lambda_wt * torch.sum(torch.abs(image_dx), 3, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.sum(torch.abs(image_dy), 3, keepdim=True)))

    smoothness_x = weights_x * (torch.abs(depth_dxdx) + torch.abs(depth_dydx)) / 2.
    smoothness_y = weights_y * (torch.abs(depth_dydy) + torch.abs(depth_dxdy)) / 2.
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


def compute_reconstr_loss(warped, ref, mask, simple=True):
    if simple:
        return F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
    else:
        alpha = 0.5
        ref_dx, ref_dy = gradient(ref * mask)
        warped_dx, warped_dy = gradient(warped * mask)
        photo_loss = F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
        grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='mean') + \
                    F.smooth_l1_loss(warped_dy, ref_dy, reduction='mean')
        return (1 - alpha) * photo_loss + alpha * grad_loss


def compute_reconstr_loss_fixed(warped, ref, mask):
    alpha = 0.5
    ref_dx, ref_dy = gradient(ref * mask, append_zeros=True)
    warped_dx, warped_dy = gradient(warped * mask, append_zeros=True)
    photo_loss = torch.sum(F.smooth_l1_loss(warped*mask, ref*mask, reduction='none'), dim=-1, keepdim=True)
    grad_loss = torch.sum(F.smooth_l1_loss(warped_dx, ref_dx, reduction='none'), dim=-1, keepdim=True) + \
                torch.sum(F.smooth_l1_loss(warped_dy, ref_dy, reduction='none'), dim=-1, keepdim=True)
    return (1 - alpha) * photo_loss + alpha * grad_loss


###########################################      Weight CNN utils        ###############################################
def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 1D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, num_groups=8, norm="batch_norm", **kwargs):
        super(Conv2d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)

        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if norm == "batch_norm" else None
        self.relu = relu
        self.gn = nn.GroupNorm(num_groups, out_channels) if norm == "group_norm" else None
        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 1D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, num_groups=8, norm="batch_norm", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if norm == "batch_norm" else None
        self.gn = nn.GroupNorm(num_groups, out_channels) if norm == "group_norm" else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class WeightNet(nn.Module):
    def __init__(self, in_channels, base_channels, out_channels=1):
        super(WeightNet, self).__init__()
        self.conv0 = Conv2d(in_channels, base_channels, padding=1)

        self.conv1 = Conv2d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv2d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv2d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv2d(base_channels * 4, base_channels * 4, padding=1)

        self.out = nn.Conv2d(base_channels * 4, out_channels, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)                   # b, 8, 128, 160
        x = self.conv2(self.conv1(conv0))       # b, 16, 64, 80
        x = self.conv4(self.conv3(x))           # b, 32, 32, 40
        x = self.out(x)                         # b, 1, 32, 40
        return x
