import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *
from typing import List, Union
from collections import OrderedDict

Align_Corners_Range = False


class ListModule(nn.Module):
    def __init__(self, modules: Union[List, OrderedDict]):
        super(ListModule, self).__init__()
        if isinstance(modules, OrderedDict):
            iterable = modules.items()
        elif isinstance(modules, list):
            iterable = enumerate(modules)
        else:
            raise TypeError('modules should be OrderedDict or List.')
        for name, module in iterable:
            if not isinstance(module, nn.Module):
                module = ListModule(module)
            if not isinstance(name, str):
                name = str(name)
            self.add_module(name, module)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


def entropy(volume, dim, keepdim=False):
    return torch.sum(-volume * volume.clamp(1e-9, 1.).log(), dim=dim, keepdim=keepdim)


class PixelwiseNet(nn.Module):
    def __init__(self):
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=1, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1):
        """forward.

        :param x1: [B, 1, D, H, W]
        """

        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1) # [B, D, H, W]
        output = self.output(x1)
        output = torch.max(output, dim=1, keepdim=True)[0] # [B, 1, H ,W]

        return output


class PixelwiseNet_ref(nn.Module):
    def __init__(self, ref_channels=32):
        super(PixelwiseNet_ref, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=1, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0)

        self.conv3 = ConvBnReLU(8+ref_channels, 64, kernel_size=3, stride=1, pad=1)
        self.conv4 = ConvBnReLU(64, 64, kernel_size=3, stride=1, pad=1)
        self.conv5 = ConvBnReLU(64, 1, kernel_size=3, stride=1, pad=1)

        self.output = nn.Sigmoid()

    def forward(self, v, r):
        """forward.

        :param v: [B, 1, D, H, W]
        :param r: [B, ref_channels, H, W]
        """

        v = self.conv2(self.conv1(self.conv0(v)))    # [B, 8, D, H, W]
        vmax = torch.max(v, dim=2)[0]                # [B, 8, H, W]
        x = torch.cat((r, vmax), dim=1)
        x = self.conv5(self.conv4(self.conv3(x)))
        output = self.output(x)

        return output


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, imgs,
                prob_volume_init=None, G=1, depth_method='regression', **kwargs):
        outputs = {}

        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shapep[1], num_depth)
        num_views = len(features)
        B = imgs.shape[0]
        _,C,H,W = features[0].shape
        ref_feature, src_features = features[0], features[1:]

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        ref_volume = ref_volume.view(B, G, C // G, num_depth, H, W)
        volume_sum = 0

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            # warped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            warped_volume = warped_volume.view(B, G, C // G, num_depth, H, W)
            volume_sum += torch.mean(ref_volume * warped_volume, dim=2)      # group-wise correlation
            del warped_volume

        volume_precost = volume_sum / (num_views - 1)

        # step 3. cost volume regularization
        cost_reg, vol_feat = cost_regularization(volume_precost, return_vol_feature=True)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        if depth_method == 'regression':
            depth = depth_regression(prob_volume, depth_values=depth_values)
        elif depth_method == 'wta':
            depth = depth_wta(prob_volume, depth_values=depth_values)
        else:
            raise NotImplementedError

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                                stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                                  dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        outputs.update(**{
            "depth": depth,
            "photometric_confidence": photometric_confidence,
            'depth_values': depth_values,
            "prob_volume": prob_volume
        })
        return outputs


class DepthNet_weighted(nn.Module):
    def __init__(self):
        super(DepthNet_weighted, self).__init__()
        self.pixel_wise_net = PixelwiseNet_ref()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, imgs,
                prob_volume_init=None, G=1, view_weights=None, depth_method='regression'):
        outputs = {}

        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shapep[1], num_depth)
        B = imgs.shape[0]
        _,C,H,W = features[0].shape
        ref_feature, src_features = features[0], features[1:]

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        ref_volume = ref_volume.view(B, G, C // G, num_depth, H, W)
        volume_sum = 0
        pixel_wise_weight_sum = 1e-5

        if view_weights == None:
            view_weight_list = []

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            warped_volume = warped_volume.view(B, G, C // G, num_depth, H, W)
            group_sim = torch.sum(ref_volume * warped_volume, dim=2)
            if view_weights == None:
                sim = torch.sum(group_sim, dim=1, keepdim=True) / C
                view_weight = self.pixel_wise_net(sim, ref_feature)
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            group_sim = group_sim / (C // G)
            view_weight = view_weight.unsqueeze(1)

            volume_sum += group_sim * view_weight
            pixel_wise_weight_sum += view_weight
            del warped_volume

        volume_precost = volume_sum.div_(pixel_wise_weight_sum)

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_precost)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        if depth_method == 'regression':
            depth = depth_regression(prob_volume, depth_values=depth_values)
        elif depth_method == 'wta':
            depth = depth_wta(prob_volume, depth_values=depth_values)
        else:
            raise NotImplementedError

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                                stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                                  dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        if view_weights is None:
            view_weights = torch.cat(view_weight_list, dim=1)  # [B, V-1, H, W]

        outputs.update(**{"depth": depth,
                          "photometric_confidence": photometric_confidence,
                          "depth_values": depth_values,
                          "prob_volume": prob_volume,
                          "view_weights": view_weights})
        return outputs


class CascadeMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8], ngroups=[1, 1, 1],
                 model_type='weighted', depth_method='regression'):
        super(CascadeMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.ngroups = ngroups
        self.model_type = model_type
        self.depth_method = depth_method
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(
            ndepths, depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1": {
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }
        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)

        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([
                CostRegNet(in_channels=self.feature.out_channels[i] if ngroups[i] == 1 else ngroups[i],
                           base_channels=self.cr_base_chs[i]) for i in range(self.num_stage)])

        if self.refine:
            self.refine_network = RefineNet()

        if model_type == 'normal':
            self.DepthNet = DepthNet()
        elif 'weighted' in model_type:
            self.DepthNet = DepthNet_weighted()
        else:
            raise NotImplementedError

    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        outputs = {}

        depth, cur_depth, view_weights = None, None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            # stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                          [img.shape[2], img.shape[3]], mode='bilinear',
                                          align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            if view_weights is not None:
                view_weights = F.interpolate(view_weights, features_stage[0].shape[-2:], mode="nearest")

            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                          ndepth=self.ndepths[stage_idx],
                                                          depth_inteval_pixel=self.depth_interals_ratio[
                                                                                  stage_idx] * depth_interval,
                                                          dtype=img[0].dtype,
                                                          device=img[0].device,
                                                          shape=[img.shape[0], img.shape[2], img.shape[3]],
                                                          max_depth=depth_max,
                                                          min_depth=depth_min)

            outputs_stage = self.DepthNet(features_stage, proj_matrices_stage,
                                          depth_values=F.interpolate(depth_range_samples.unsqueeze(1),
                                                                     [self.ndepths[stage_idx],
                                                                      img.shape[2] // int(stage_scale),
                                                                      img.shape[3] // int(stage_scale)],
                                                                     mode='trilinear',
                                                                     align_corners=Align_Corners_Range).squeeze(1),
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else
                                          self.cost_regularization[stage_idx],imgs=imgs, G=self.ngroups[stage_idx],
                                          view_weights=view_weights)

            depth = outputs_stage['depth']
            view_weights = outputs_stage.get('view_weights', None)

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)
        # depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs
