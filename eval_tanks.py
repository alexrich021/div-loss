import argparse
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
import errno
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image

from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import save_pfm
from models.casmvsnet import CascadeMVSNet
from torchvision import transforms
from filter.tank_test_config import tank_cfg
from filter import dypcd, normal

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='weighted', choices=['normal', 'weighted'])

parser.add_argument('--dataset', default='general_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testlist', default='all')
parser.add_argument('--split', default='intermediate', help='select data')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')

parser.add_argument('--loadckpt', default='./outputs/DIV-MVS/model_000014.ckpt', help='load prestrained model')
parser.add_argument('--outdir', default='./outputs/DIV-MVS/tanks', help='output dir for depth maps')
parser.add_argument('--plydir', default='./outputs/DIV-MVS/tanks_ply', help='output dir of fusion points')

parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--ngroups', type=str, default="8,4,2", help='ngroups')
parser.add_argument('--depth_inter_r', type=str, default="3,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--num_view', type=int, default=11, help='num of view')
parser.add_argument('--max_h', type=int, default=1056, help='testing max h')
parser.add_argument('--max_w', type=int, default=1920, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=4, help='depth_filter worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')
parser.add_argument('--no_depth_pred', action='store_true')

parser.add_argument('--filter', type=str, default='normal', choices=['normal', 'dypcd'])

# dypcd filter settings
parser.add_argument('--dist_base', type=float, default=1 / 4)
parser.add_argument('--rel_diff_base', type=float, default=1 / 1300)

# model setting
parser.add_argument('--dcn', type=bool, default=False, help="use deformable convolution in 2D backbone")
parser.add_argument('--gn', default=False, help='apply group normalization.')
parser.add_argument('--true_gpu', default="1", help='using true gpu')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.true_gpu


# read an image
def read_img(filename, img_wh):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)

    return np_img


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))

    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[1])

    return intrinsics, extrinsics, depth_min, depth_max


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def save_depth_img(filename, depth):
    # assert mask.dtype == np.bool
    depth = depth * 255
    depth = depth.astype(np.uint8)
    Image.fromarray(depth).save(filename)


def center_image(imgs):
    """ normalize image input """
    var = torch.var(imgs, dim=(1, 2), keepdims=True)
    mean = torch.mean(imgs, dim=(1, 2), keepdim=True)
    ctr_imgs = (imgs - mean) / (torch.sqrt(var) + 0.00000001)
    return ctr_imgs


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) != 0:
                data.append((ref_view, src_views))
    return data


def write_depth_img_2(filename, depth):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)
    im.save(filename)


# run MVS model to save depth maps
def save_depth(datapath, scan, img_wh=(1920, 1056)):
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(datapath, [scan], 'test', args.num_view, args.numdepth, max_h=img_wh[1], max_w=img_wh[0],
                              fix_res=args.fix_res)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method,
                          ngroups=[int(g) for g in args.ngroups.split(',') if g],
                          model_type=args.model)

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    num_stage = 3

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            sample_cuda = tocuda(sample)

            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

            outputs = tensor2numpy(outputs)
            del sample_cuda
            print('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), time.time() - start_time))
            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"]

            # save depth maps and confidence maps
            for filename, cam, img, depth_est, depth2, depth1, photometric_confidence, pc2, pc1 \
                    in zip(filenames, cams, imgs, outputs["depth"],
                           outputs["stage2"]["depth"],
                           outputs["stage1"]["depth"],
                           outputs["photometric_confidence"],
                           outputs["stage2"]["photometric_confidence"],
                           outputs["stage1"]["photometric_confidence"]):
                depth_filename2 = os.path.join(args.outdir, filename.format('depth_est', '_stage2.pfm'))
                depth_filename1 = os.path.join(args.outdir, filename.format('depth_est', '_stage1.pfm'))

                h, w = photometric_confidence.shape
                pc2 = cv2.resize(pc2, (w, h), interpolation=cv2.INTER_NEAREST)
                pc1 = cv2.resize(pc1, (w, h), interpolation=cv2.INTER_NEAREST)
                confidence_filename2 = os.path.join(args.outdir, filename.format('confidence', '_stage2.pfm'))
                confidence_filename1 = os.path.join(args.outdir, filename.format('confidence', '_stage1.pfm'))

                img = inv_normalize(img[0]).numpy()  # ref view
                cam = cam[0]  # ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                depth_color_filename = os.path.join(args.outdir, filename.format('depth_map', '.jpg'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(depth_color_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)

                # save depth maps
                save_pfm(depth_filename, depth_est)
                save_pfm(depth_filename2, depth2)
                save_pfm(depth_filename1, depth1)
                cv2.imwrite(depth_color_filename, visualize_depth_2(depth_est))

                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                save_pfm(confidence_filename2, pc2)
                save_pfm(confidence_filename1, pc1)

                # save cams, img
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)


if __name__ == '__main__':
    # intermediate dataset
    if args.split == "intermediate":
        scans = [
            'Family',
            'Francis',
            'Horse',
            'Lighthouse',
            'M60',
            'Panther',
            'Playground',
            'Train'
        ]
    # advanced dataset
    elif args.split == "advanced":
        scans = [
            'Auditorium',
            'Ballroom',
            'Courtroom',
            'Museum',
            'Palace',
            'Temple'
        ]
    else:
        raise NotImplementedError

    datapath = os.path.join(args.testpath, args.split)
    if not args.no_depth_pred:
        for scan in scans:
            scene_cfg = getattr(tank_cfg, scan)
            save_depth(datapath, scan, (scene_cfg.max_w, scene_cfg.max_h))

    if args.filter == 'normal':
        normal.normal_filter(args, scans)
    elif args.filter == 'dypcd':
        dypcd.dypcd_filter(args, scans, args.num_worker)
    else:
        raise NotImplementedError
