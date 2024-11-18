from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
import torch
import torch.nn.functional as F


def compute_K_inv(K):
    fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    K_inv = torch.zeros_like(K)
    K_inv[:, 0, 0] = 1. / fx
    K_inv[:, 1, 1] = 1. / fy
    K_inv[:, 2, 2] = 1.
    K_inv[:, 0, 2] = -cx / fx
    K_inv[:, 1, 2] = -cy / fy
    return K_inv


def depth2meshvals(depth, cam):
    B, H, W = depth.shape
    K_inv = compute_K_inv(cam[:, 1, :3, :3])
    RT = cam[:, 0, :3, :3].permute(0, 2, 1)
    t = cam[:, 0, :3, 3:]

    xx = torch.matmul(
        torch.ones([H, 1]).type_as(depth),
        torch.arange(W).unsqueeze(0).type_as(depth)
    )  # [height, width]
    yy = torch.matmul(
        torch.arange(H).unsqueeze(1).type_as(depth),
        torch.ones([1, W]).type_as(depth)
    )
    zz = torch.ones_like(xx)
    pts = torch.stack((xx, yy, zz), dim=0).view(3, H*W).unsqueeze(0).repeat(B, 1, 1)    # B, 3, H*W

    depth_pts = torch.bmm(RT, torch.bmm(K_inv, pts * depth.view(B, 1, H*W)) - t).permute(0, 2, 1)   # B, H*W, 3

    # get triangle indices
    xx = xx.long()
    yy = yy.long()
    tris1_x = torch.stack((xx[:-1, :-1], xx[1:, :-1], xx[:-1, 1:]), dim=2)
    tris1_y = torch.stack((yy[:-1, :-1], yy[1:, :-1], yy[:-1, 1:]), dim=2)
    tris1 = tris1_x + tris1_y * W

    tris2_x = torch.stack((xx[:-1, 1:], xx[1:, :-1], xx[1:, 1:]), dim=2)
    tris2_y = torch.stack((yy[:-1, 1:], yy[1:, :-1], yy[1:, 1:]), dim=2)
    tris2 = tris2_x + tris2_y * W

    mask = depth > 0.
    max_tris = (H-1) * (W-1) * 2
    tris_list = []
    for b in range(B):
        valid1 = mask[b, :-1, :-1] & mask[b, 1:, :-1] & mask[b, :-1, 1:]
        valid2 = mask[b, :-1, 1:] & mask[b, 1:, :-1] & mask[b, 1:, 1:]
        tris_masked = torch.cat((tris1[valid1], tris2[valid2]), dim=0)

        npack = max_tris - tris_masked.shape[0]
        neg_ones = -torch.ones((npack, 3), dtype=torch.long, device=tris_masked.device)
        tris_packed = torch.cat((tris_masked, neg_ones), dim=0)

        tris_list.append(tris_packed)
    tris = torch.stack(tris_list, dim=0)
    return depth_pts, tris


def check_src_occlusion(depth, cams, tol=2.0, return_zbuffer=False):
    with torch.no_grad():
        ref_cam, src_cams = cams[:, 0], cams[:, 1:]
        verts, tris = depth2meshvals(depth, ref_cam)

        nverts = verts.shape[1]
        ntris = tris.shape[1]
        B, H, W = depth.shape
        V = src_cams.shape[1]
        verts = verts.unsqueeze(1).expand(B, V, nverts, 3).reshape(B*V, nverts, 3)
        tris = tris.unsqueeze(1).expand(B, V, ntris, 3).reshape(B*V, ntris, 3)
        meshes = Meshes(verts=verts, faces=tris)

        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=64_000,
            cull_backfaces=False,
        )

        # convert to pytorch3d camera coordinate system
        src_cams = src_cams.reshape(B*V, 2, 4, 4)
        fx, fy = src_cams[:, 1, 0, 0].tolist(), src_cams[:, 1, 1, 1].tolist()
        px, py = (W - src_cams[:, 1, 0, 2]).tolist(), (H - src_cams[:, 1, 1, 2]).tolist()
        RT = src_cams[:, 0, :3, :3].permute(0, 2, 1)
        t = src_cams[:, 0, :3, 3]
        cameras = PerspectiveCameras(focal_length=tuple(zip(fx, fy)),
                                     principal_point=tuple(zip(px, py)),
                                     in_ndc=False,
                                     image_size=((H, W), ),
                                     R=RT, T=t).cuda()

        # render z buffer
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )
        fragments = rasterizer(meshes)
        z_buffer = fragments.zbuf.view(B*V, 1, H, W).flip(dims=(2, 3))  # pytorch3d uses inverse direction for image space

        # now render verts (i.e. back-projected reference depth) to each src image and check against z buffer
        K_src = src_cams[:, 1, :3, :3]
        R_src = src_cams[:, 0, :3, :3]
        KR_src = torch.bmm(K_src, R_src)
        t_src = src_cams[:, 0, :3, 3:]
        Kt_src = torch.bmm(K_src, t_src)

        pts_src = torch.bmm(verts, KR_src.permute(0, 2, 1)) + Kt_src.view(B*V, 1, 3)
        z_src = pts_src[..., 2:]
        grid = pts_src[..., :2] / z_src
        grid = grid.view(B*V, H, W, 2)
        grid[..., 0] = 2 * (grid[..., 0] / float(W - 1)) - 1
        grid[..., 1] = 2 * (grid[..., 1] / float(H - 1)) - 1
        z_samples = F.grid_sample(z_buffer, grid, mode='bilinear', padding_mode='border')

        z_buffer = z_buffer.view(B, V, H, W)
        z_samples = z_samples.view(B, V, H, W)
        z_src = z_src.view(B, V, H, W)

        occ_mask = ((z_src - z_samples) < tol) | (z_samples == -1)
    if return_zbuffer:
        return occ_mask, z_buffer
    else:
        return occ_mask