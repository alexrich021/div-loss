import argparse, os, sys, time, gc, datetime

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import find_dataset_def
from models import *
from models import CascadeMVSNet
from utils import *
from losses.unsup_loss import UnsupLossMultiStage, UnsupLossMultiStage_learned
from losses.aug_loss import random_image_mask, AugLossMultiStage

cudnn.benchmark = True

# arguments
parser = argparse.ArgumentParser(description='A PyTorch Implementation of IL-Base training')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test'])
parser.add_argument('--model', default='weighted', help='select model', choices=['weighted', 'normal'])
parser.add_argument('--device', default='cuda', help='select model')
parser.add_argument('--ddp_backend', default='nccl', type=str, choices=['nccl', 'Gloo'],
                    help='Determines backend to use with DDP training. If training script hangs on start, try '
                         'switching backends')

parser.add_argument('--dataset', default='dtu_train', help='select dataset')
parser.add_argument('--test_dataset', default='dtu_yao', help='select test dataset')
parser.add_argument('--trainpath', help='train datapath', default=os.path.join(os.path.expanduser('~'), 'dtu'))
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', default="./lists/dtu/train.txt", help='train list')
parser.add_argument('--testlist', default='./lists/dtu/test.txt', help='test list')

parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2",
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
parser.add_argument('--num_view', type=int, default=3, help='the number of source views')

parser.add_argument('--logdir', default='outputs/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--loadckpt', type=str, help='Path to cas checkpoint for testing')

parser.add_argument('--summary_freq', type=int, default=40, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--random_seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--ngroups', type=str, default="8,4,2", help='ngroups')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
parser.add_argument('--w_aug', type=float, default=0.01, help='weight of aug loss')
parser.add_argument('--loss_type', type=str, default='learned_occ', choices=['topk', 'topk_occ', 'learned', 'learned_occ'])
parser.add_argument('--supervision_view_selection', type=str, default='score', choices=['topk', 'score'])
parser.add_argument('--reconstr_loss_type', type=str, default='smooth_l1', choices=['smooth_l1', 'smooth_l1_fixed'])
parser.add_argument('--mesh_render_tol', type=float, default=2.0,
                    help='Set error tolerance for determining occlusion when rendering mesh')
parser.add_argument('--smoothness_type', type=str, default='2nd_order', choices=['1st_order', '2nd_order'])
parser.add_argument('--smoothness_clip', type=float, default=2.0, help='Clipping value when using 2nd_order smoothness')
parser.add_argument('--smoothness_lambda', type=float, default=0.36, help='Weight for smoothness loss')

# distributed training args
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument("--local_rank", type=int, default=0)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1


# main function
def train(model, model_loss, aug_loss, test_model_loss, optimizer, TrainImgLoader, TestImgLoader, train_sampler,
          test_sampler, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0 / 3, warmup_iters=500,
                                     last_epoch=len(TrainImgLoader) * start_epoch - 1)

    logger = SummaryWriter(args.logdir)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx
        avg_train_scalars = DictAverageMeter()
        avg_aug_scalars = DictAverageMeter()

        if is_distributed:
            train_sampler.set_epoch(epoch_idx)
            test_sampler.set_epoch(epoch_idx)

        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            sample_cuda = tocuda(sample)
            optimizer.zero_grad()

            # stage 1: standard self-supervision loss update
            loss, scalar_outputs, image_outputs, pseudo_depth, backbone_outputs, \
                loss_base = train_sample(model, model_loss, sample_cuda, args, epoch_idx)
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                    print("Epoch {}/{}, Iter-S1 {}/{}, lr {:.6f}, train loss = {:.3f},  depth loss = {:.3f}, "
                          "time = {:.3f}".format(epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                                                 optimizer.param_groups[0]["lr"], loss,
                                                 scalar_outputs['depth_loss_stage3'], time.time() - start_time))
                avg_train_scalars.update(scalar_outputs)

            # stage 2: augmentation self-supervision loss update
            loss_t, aug_scalar_outputs, aug_image_outputs, loss_aug = train_sample_aug(
                model, aug_loss, sample_cuda, args, pseudo_depth, epoch_idx)
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', aug_scalar_outputs, global_step)
                    save_images(logger, 'train', aug_image_outputs, global_step)
                    print("Epoch {}/{}, Iter-S2 {}/{}, lr {:.6f}, aug loss = {:.3f}, depth loss = {:.3f}, "
                          "time = {:.3f}".format(
                            epoch_idx, args.epochs, batch_idx, len(TrainImgLoader), optimizer.param_groups[0]["lr"],
                            loss_t, aug_scalar_outputs['aug_loss_stage3'], time.time() - start_time))
                avg_aug_scalars.update(aug_scalar_outputs)

            loss_full = loss_base + loss_aug
            loss_full.backward()
            optimizer.step()

            lr_scheduler.step()

            del scalar_outputs, image_outputs, aug_scalar_outputs, aug_image_outputs

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            save_scalars(logger, 'fulltrain', avg_train_scalars.mean(), global_step)
            save_scalars(logger, 'fulltrain', avg_aug_scalars.mean(), global_step)
            print("avg_train_scalars:", avg_train_scalars.mean())
            print("avg_aug_scalars:", avg_aug_scalars.mean())
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
                if 'learned' in args.loss_type:
                    torch.save({'model': model_loss.module.state_dict()},
                               "{}/model_{:0>6}_loss.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                sample_cuda = tocuda(sample)
                loss, scalar_outputs, image_outputs, backbone_outputs = test_sample_depth(model, test_model_loss, sample_cuda, args)

                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_images(logger, 'test', image_outputs, global_step)
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                            epoch_idx, args.epochs, batch_idx, len(TestImgLoader), loss,
                            scalar_outputs["depth_loss"], time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        sample_cuda = tocuda(sample)
        loss, scalar_outputs, image_outputs, backbone_outputs = test_sample_depth(model, model_loss, sample_cuda, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, sample_cuda, args, epoch_idx):
    model.train()

    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss, scalars, images = model_loss(outputs, sample_cuda["sup_imgs"], sample_cuda["sup_proj_matrices"],
                                       dlossw=[float(e) for e in args.dlossw.split(",") if e])
    if 'learned' in args.loss_type:
        loss = loss.mean()
        scalars = {k: v.mean() for k, v in scalars.items()}
    scalar_outputs = {
        "loss": loss
    }
    scalar_outputs["thres2mm_accu"] = 1.0 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_accu"] = 1.0 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_accu"] = 1.0 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
    scalar_outputs.update(**scalars)

    image_outputs = {
        "depth_est": depth_est,
        "depth_gt": depth_gt,
        "ref_img": unpreprocess(sample_cuda["imgs"][:, 0], shape=(1, 3, 1, 1)),
        "errormap": (depth_est - depth_gt).abs() * mask,
    }
    image_outputs.update(**images)

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs), \
           depth_est.detach(), outputs, loss


def adjust_w_aug(epoch_idx, w_aug):
    if epoch_idx >= 2 - 1:
        w_aug *= 2
    if epoch_idx >= 4 - 1:
        w_aug *= 2
    if epoch_idx >= 6 - 1:
        w_aug *= 2
    if epoch_idx >= 8 - 1:
        w_aug *= 2
    if epoch_idx >= 10 - 1:
        w_aug *= 2
    return w_aug


def train_sample_aug(model, aug_loss, sample_cuda, args, pseudo_depth, epoch_idx):
    model.train()

    # augmentation
    imgs_aug = sample_cuda["imgs_aug"]
    ref_img = imgs_aug[:, 0]
    ref_img, filter_mask = random_image_mask(ref_img, filter_size=(ref_img.size(2) // 3, ref_img.size(3) // 3))
    imgs_aug[:, 0] = ref_img

    outputs = model(imgs_aug, sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    dlossw = [float(e) for e in args.dlossw.split(",") if e]
    loss, scalars = aug_loss(outputs, pseudo_depth, filter_mask, dlossw=dlossw)

    # adjust w_aug
    w_aug = adjust_w_aug(epoch_idx, args.w_aug)
    loss = loss * w_aug

    scalar_outputs = {
        "aug_loss": loss,
    }
    for key in scalars.keys():
        scalar_outputs[key] = scalars[key]

    image_outputs = {
        "aug_depth_est": depth_est
    }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["aug_loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs), loss


@make_nograd_func
def test_sample_depth(model, model_loss, sample_cuda, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    scalar_outputs = {
        "loss": loss,
        "depth_loss": depth_loss,
    }
    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_accu"] = 1 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_accu"] = 1 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_accu"] = 1 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
    scalar_outputs["thres2mm_abserror"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0])

    image_outputs = {
        "depth_est": depth_est,
        "depth_gt": sample_cuda["depth"]["stage{}".format(num_stage)],
        "ref_img": unpreprocess(sample_cuda["imgs"][:, 0], shape=(1, 3, 1, 1)),
        "errormap": (depth_est - depth_gt).abs() * mask
    }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs), outputs


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None

    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend=args.ddp_backend, init_method="env://"
        )

    set_random_seed(args.seed)
    device = torch.device(args.local_rank)

    if (not is_distributed) or (dist.get_rank() == 0):
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer
    model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method,
                          ngroups=[int(g) for g in args.ngroups.split(',') if g],
                          model_type=args.model)

    # to device
    model.to(device)
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print(model)

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    compute_occ_mask = 'occ' in args.loss_type
    if 'topk' in args.loss_type:
        model_loss = UnsupLossMultiStage(args.smoothness_type, args.smoothness_lambda, args.smoothness_clip,
                                         compute_occ_mask, args.mesh_render_tol, args.reconstr_loss_type).to(device)
    elif 'learned' in args.loss_type:
        model_loss = UnsupLossMultiStage_learned(args.smoothness_type, args.smoothness_lambda, args.smoothness_clip,
                                                 args.num_view, compute_occ_mask, args.mesh_render_tol,
                                                 args.reconstr_loss_type).to(device)
        params.extend(list(filter(lambda p: p.requires_grad, model_loss.parameters())))
    else:
        raise NotImplementedError
    aug_loss = AugLossMultiStage().to(device)
    test_model_loss = cas_mvsnet_loss

    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if ("model" in fn and "loss" not in fn)]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1

        if 'learned' in args.loss_type:
            loadckpt_loss = loadckpt.replace('.ckpt', '_loss.ckpt')
            print("resuming loss", loadckpt_loss)
            state_dict_loss = torch.load(loadckpt_loss, map_location=torch.device("cpu"))
            model_loss.load_state_dict(state_dict_loss['model'])

    elif args.loadckpt is not None:
        print("resuming", args.loadckpt)
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1

        if 'learned' in args.loss_type:
            loadckpt_loss = args.loadckpt.replace('.ckpt', '_loss.ckpt')
            print("resuming loss", loadckpt_loss)
            state_dict_loss = torch.load(loadckpt_loss, map_location=torch.device("cpu"))
            model_loss.load_state_dict(state_dict_loss['model'])

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=False
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
        if 'learned' in args.loss_type:
            model_loss = torch.nn.parallel.DistributedDataParallel(
                model_loss, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=False
                # this should be removed if we update BatchNorm stats
                # broadcast_buffers=False,
            )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            if 'learned' in args.loss_type:
                model_loss = nn.DataParallel(model_loss)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.num_view + 1, args.numdepth,
                               args.interval_scale, sup_view_selection=args.supervision_view_selection)
    test_MVSDataset = find_dataset_def(args.test_dataset)
    test_dataset = test_MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale,
                                   l3_only=True)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank(), shuffle=True)
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank(), shuffle=True)

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1, drop_last=True)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False)
    else:
        train_sampler, test_sampler = None, None
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=False)

    if args.mode == "train":
        train(model, model_loss, aug_loss, test_model_loss, optimizer, TrainImgLoader, TestImgLoader, train_sampler,
              test_sampler, start_epoch, args)
    elif args.mode == "test":
        test(model, test_model_loss, TestImgLoader, args)
    else:
        raise NotImplementedError
