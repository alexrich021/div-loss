import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.modules import *
from losses.homography import *
from losses.mesh_render import check_src_occlusion


class UnSupLoss(nn.Module):
    def __init__(self, smoothness_type='1st_order', smoothness_lambda=0.18, clip=2.0, compute_occ_mask=False,
                 mesh_render_tol=2.0, reconstr_loss_type='smooth_l1'):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()
        self.compute_occ_mask = compute_occ_mask
        self.mesh_render_tol = mesh_render_tol
        self.smoothness_lambda = smoothness_lambda
        self.clip = clip
        self.depth_smoothness = get_smoothness_func(smoothness_type)
        self.reconstr_loss_func = get_reconstr_loss_func(reconstr_loss_type)

    def forward(self, imgs, cams, depth, stage_idx):
        if self.compute_occ_mask:
            occ_mask = check_src_occlusion(depth, cams, tol=self.mesh_render_tol).unsqueeze(-1)
        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        num_views = len(imgs)

        ref_img = imgs[0]

        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]

            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25, recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5, recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]

            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)

            if self.compute_occ_mask:
                mask = (mask.bool() & occ_mask[:, view-1].bool()).float()

            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = self.reconstr_loss_func(warped_img, ref_img, mask)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        self.smooth_loss += self.depth_smoothness(depth.unsqueeze(dim=-1), ref_img, clip=self.clip)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses, dim=4)
        top_vals, top_inds = torch.topk(reprojection_volume, k=1, sorted=False, largest=False, dim=4)
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))

        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + self.smoothness_lambda * self.smooth_loss
        return self.unsup_loss


class UnSupLoss_weighted(nn.Module):
    def __init__(self, smoothness_type='1st_order', smoothness_lambda=0.18, clip=2.0, compute_occ_mask=False,
                 mesh_render_tol=2.0, reconstr_loss_type='smooth_l1'):
        super(UnSupLoss_weighted, self).__init__()
        self.ssim = SSIM()
        self.compute_occ_mask = compute_occ_mask
        self.mesh_render_tol = mesh_render_tol
        self.smoothness_type = smoothness_type
        self.smoothness_lamda = smoothness_lambda
        self.clip = clip
        self.depth_smoothness = get_smoothness_func(smoothness_type)
        self.reconstr_loss_func = get_reconstr_loss_func(reconstr_loss_type)

    def forward(self, imgs, cams, depth, stage_idx, weights):
        B, V, _, H, W = imgs.shape
        if self.compute_occ_mask:
            occ_mask = check_src_occlusion(depth, cams, tol=self.mesh_render_tol).unsqueeze(-1)

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"

        num_views = len(imgs)

        ref_img = imgs[0]

        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []

        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]

            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25, recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5, recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]

            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)

            if self.compute_occ_mask:
                mask = (mask.bool() & occ_mask[:, view-1].bool()).float()

            warped_img_list.append(warped_img)
            mask_list.append(mask)

        masks = torch.stack(mask_list, dim=1)   # [B, V, H, W, 1]
        mask_final = (torch.sum(masks, dim=1) > 0.).float()

        if stage_idx != 2:
            B, V, H, W = weights.shape[:4]
            weights = F.interpolate(weights.view(B * V, 1, H, W), (depth.shape[-2:]), mode='bilinear')\
                .view(B, V, *depth.shape[-2:], 1)

        weights_final = (weights * masks) + 1e-5
        weights_final = weights_final / torch.sum(weights_final, dim=1, keepdim=True)

        warped_imgs = torch.stack(warped_img_list, dim=1)    # [B, V, H, W, 3]
        warped_img_final = torch.sum(weights_final * warped_imgs, dim=1)

        self.reconstr_loss = self.reconstr_loss_func(warped_img_final, ref_img, mask_final)
        self.ssim_loss = 2 * torch.mean(self.ssim(ref_img, warped_img_final, mask_final))
        self.smooth_loss = self.depth_smoothness(depth.unsqueeze(dim=-1), ref_img, lambda_wt=1.0, clip=self.clip)

        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + self.smoothness_lamda * self.smooth_loss
        return self.unsup_loss, warped_img_final, warped_imgs[:, 0]


class UnSupLoss_learned(nn.Module):
    def __init__(self, smoothness_type='1st_order', smoothness_lambda=0.18, clip=2.0, num_src_views=3,
                 compute_occ_mask=False, mesh_render_tol=2.0, reconstr_loss_type='smooth_l1'):
        super(UnSupLoss_learned, self).__init__()
        self.ssim = SSIM()
        self.weight_net = WeightNet(num_src_views*3, 16, out_channels=num_src_views)
        self.view_weights = 0.
        self.compute_occ_mask = compute_occ_mask
        self.mesh_render_tol = mesh_render_tol
        self.masks = None
        self.smoothness_type = smoothness_type
        self.smoothness_lambda = smoothness_lambda
        self.clip = clip
        self.depth_smoothness = get_smoothness_func(smoothness_type)
        self.reconstr_loss_func = get_reconstr_loss_func(reconstr_loss_type)

    def forward(self, imgs, cams, depth, stage_idx):
        B, V, _, H, W = imgs.shape
        if self.compute_occ_mask:
            occ_mask = check_src_occlusion(depth, cams, tol=self.mesh_render_tol).unsqueeze(-1)

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"

        num_views = len(imgs)

        ref_img = imgs[0]

        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []

        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]

            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25, recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5, recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]

            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)

            if self.compute_occ_mask:
                mask = (mask.bool() & occ_mask[:, view-1].bool()).float()

            warped_img_list.append(warped_img)
            mask_list.append(mask)

        warped_imgs = torch.stack(warped_img_list, dim=1)    # [B, V, H, W, 3]
        masks = torch.stack(mask_list, dim=1)   # [B, V, H, W, 1]
        self.masks = masks
        mask_final = (torch.sum(masks, dim=1) > 0.).float()

        x = torch.cat(warped_img_list, dim=-1)
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        w_out = self.weight_net(x)
        w_out = F.interpolate(w_out, (H, W), mode='bilinear')
        weights = w_out.unsqueeze(4)    # [B, V, H, W, 1]
        weights = F.softmax(weights, dim=1)

        weights_final = (weights * masks) + 1e-5
        weights_final = weights_final / torch.sum(weights_final, dim=1, keepdim=True)
        warped_img_final = torch.sum(weights_final * warped_imgs, dim=1)

        self.view_weights = weights
        if self.compute_occ_mask:
            self.occ_mask = occ_mask.float()
        self.reconstr_loss = self.reconstr_loss_func(warped_img_final, ref_img, mask_final)
        self.ssim_loss = 2 * torch.mean(self.ssim(ref_img, warped_img_final, mask_final))
        self.smooth_loss = self.depth_smoothness(depth.unsqueeze(dim=-1), ref_img, lambda_wt=1.0, clip=self.clip)

        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + self.smoothness_lambda * self.smooth_loss
        return self.unsup_loss, warped_img_final, warped_imgs[:, 0]


class UnsupLossMultiStage(nn.Module):
    def __init__(self, smoothness_type='1st_order', smoothness_lambda=0.18, clip=2.0, compute_occ_mask=False,
                 mesh_render_tol=2.0, reconstr_loss_type='smooth_l1'):
        super(UnsupLossMultiStage, self).__init__()
        self.unsup_loss = UnSupLoss(smoothness_type, smoothness_lambda, clip, compute_occ_mask, mesh_render_tol,
                                    reconstr_loss_type)

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        image_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx, stage_inputs['features'])

            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs, image_outputs


class UnsupLossMultiStage_learned(nn.Module):
    def __init__(self, smoothness_type='1st_order', smoothness_lambda=0.18, clip=2.0, num_src_views=3,
                 compute_occ_mask=False, mesh_render_tol=2.0, reconstr_loss_type='smooth_l1'):
        super(UnsupLossMultiStage_learned, self).__init__()
        self.smoothness_type = smoothness_type
        self.unsup_loss_learned = UnSupLoss_learned(smoothness_type, smoothness_lambda, clip, num_src_views,
                                                    compute_occ_mask, mesh_render_tol, reconstr_loss_type)
        self.unsup_loss = UnSupLoss_weighted(smoothness_type, smoothness_lambda, clip, compute_occ_mask,
                                             mesh_render_tol, reconstr_loss_type)

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        image_outputs = {}
        stage_idx_learned = 2

        # first, do learned loss for weights
        stage_key = f'stage{stage_idx_learned+1}'
        stage_inputs = inputs[stage_key]
        depth_est = stage_inputs['depth']
        depth_loss, final_img, src_warp = self.unsup_loss_learned(imgs, cams[stage_key], depth_est, stage_idx_learned)

        # tensorboard logging
        scalar_outputs["depth_loss_{}".format(stage_key)] = depth_loss
        scalar_outputs["reconstr_loss_{}".format(stage_key)] = self.unsup_loss_learned.reconstr_loss
        scalar_outputs["ssim_loss_{}".format(stage_key)] = self.unsup_loss_learned.ssim_loss
        scalar_outputs["smooth_loss_{}".format(stage_key)] = self.unsup_loss_learned.smooth_loss
        image_outputs['weighted_warp'] = final_img.permute(0, 3, 1, 2)
        image_outputs['src_warp'] = src_warp.permute(0, 3, 1, 2)
        if self.unsup_loss_learned.compute_occ_mask:
            image_outputs['occ_mask'] = self.unsup_loss_learned.occ_mask[:, 0].permute(0, 3, 1, 2)

        if depth_loss_weights is not None:
            total_loss += depth_loss_weights[stage_idx_learned] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

        view_weights = self.unsup_loss_learned.view_weights

        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            if stage_idx == stage_idx_learned:
                continue

            depth_est = stage_inputs["depth"]
            depth_loss, final_img, src_warp = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx, view_weights)

            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            # tensorboard logging
            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs, image_outputs
