import argparse
import glob
import os
import numpy as np
import cv2
import tqdm
from multiprocessing import Pool
from functools import partial
import signal
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', default='outputs/DIV-MVS/scannetpp',
                    help='Directory containing ScanNet++ predicted point clouds')
parser.add_argument('--testpath', default=os.path.join(os.path.expanduser('~'), 'scannetpp_mvs'),
                    help='Path to pre-processed ScanNet++ dataset')
parser.add_argument('--fscore_thresh', type=float, default=0.01,
                    help='F-score distance threshold, in meters')
parser.add_argument('--downsample_dist', type=float, default=0.005,
                    help='Voxel side length for voxelizing both the predicted and ground truth reconstructions')
parser.add_argument('--num_worker', type=int, default=2, help='Number of worker threads')

args = parser.parse_args()


def eval_pcd(pcd_pred, pcd_trgt, threshold=0.05):
    """ Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal

    Returns:
        Dict of mesh metrics
    """

    _, dist1 = nn_correspondance(pcd_trgt, pcd_pred)
    _, dist2 = nn_correspondance(pcd_pred, pcd_trgt)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist1 < threshold).astype('float'))
    recal = np.mean((dist2 < threshold).astype('float'))
    fscore = 2 * precision * recal / max(precision + recal, 1e-8)

    acc = np.mean(dist1)
    comp = np.mean(dist2)
    chamfer = (acc + comp) / 2.

    metrics = {
        'acc': np.mean(dist1),
        'comp': np.mean(dist2),
        'chamfer': chamfer,
        'prec': precision,
        'recal': recal,
        'fscore': fscore,
    }
    return metrics


def nn_correspondance(pcd1, pcd2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(pcd1.points) == 0 or len(pcd2.points) == 0:
        return indices, distances
    kdtree = o3d.geometry.KDTreeFlann(pcd1)

    for vert in np.asarray(pcd2.points):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    intrinsics[:2, :] /= 4.0
    return intrinsics, extrinsics


def load_gt(scene, args):
    gt_file = os.path.join(args.testpath, scene, 'mesh_aligned_0.05.ply')
    gt_mesh = o3d.io.read_triangle_mesh(gt_file)
    gt_vox = o3d.geometry.VoxelGrid.create_from_triangle_mesh(gt_mesh, voxel_size=args.downsample_dist)
    gt_pts = np.asarray([gt_vox.origin + pt.grid_index * gt_vox.voxel_size for pt in gt_vox.get_voxels()])
    gt_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(gt_pts))
    return gt_mesh, gt_pcd


def trim_gt(scene, args, gt_pcd):
    pts = np.asarray(gt_pcd.points)
    mask = np.zeros(pts.shape[0], dtype=bool)

    # determine image height/width
    depth_file = glob.glob(os.path.join(args.pred_dir, scene, 'depth_map', '*.jpg'))[0]
    H, W, = cv2.imread(depth_file).shape[:2]

    cams_dir = os.path.join(args.pred_dir, scene, 'cams')
    cam_files = glob.glob(os.path.join(cams_dir, '*_cam.txt'))
    for cam_file in tqdm.tqdm(cam_files, total=len(cam_files)):
        K, P = read_cam_file(cam_file)
        pts_cam = P[:3, :3] @ pts.T + P[:3, 3:]
        pts_img = K @ pts_cam
        pts_xy = pts_img / pts_img[-1:]
        x, y = pts_xy[0], pts_xy[1]
        z = pts_img[2]
        valid_x = (x >= 0) & (x <= (W - 1))
        valid_y = (y >= 0) & (y <= (H - 1))
        valid_z = z > 0.001
        valid = valid_x & valid_y & valid_z
        mask = mask | valid
    pts_trimmed = pts[mask]
    pcd_trimmed = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_trimmed))
    return pcd_trimmed


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def eval_worker(scene):
    metrics_file = os.path.join(results_dir, f'{scene}.txt')
    if os.path.exists(metrics_file):
        print(f'{scene} metrics found, skipping...')
        return

    gt_mesh, gt_pcd = load_gt(scene, args)
    pred_pcd = o3d.io.read_point_cloud(os.path.join(args.pred_dir, f'{scene}.ply'))
    pred_pcd_dwn = pred_pcd.voxel_down_sample(args.downsample_dist)
    metrics = eval_pcd(pred_pcd_dwn, gt_pcd, args.fscore_thresh)

    print(f'{scene} | {metrics["prec"]:.4f} | {metrics["recal"]:.4f} | {metrics["fscore"]:.4f}')

    with open(metrics_file, 'w') as f:
        f.write(f'{metrics["acc"]:.6f}')
        f.write('\n')
        f.write(f'{metrics["comp"]:.6f}')
        f.write('\n')
        f.write(f'{metrics["chamfer"]:.6f}')
        f.write('\n')
        f.write(f'{metrics["prec"]:.6f}')
        f.write('\n')
        f.write(f'{metrics["recal"]:.6f}')
        f.write('\n')
        f.write(f'{metrics["fscore"]:.6f}')


scenes = sorted([os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.pred_dir, '*.ply'))])
results_dir = os.path.join(args.pred_dir, f'results_{args.fscore_thresh:.2f}')
os.makedirs(results_dir, exist_ok=True)

# compute per-scene metrics
partial_func = partial(eval_worker)
p = Pool(args.num_worker, init_worker)
try:
    p.map(partial_func, scenes)
except KeyboardInterrupt:
    print("....\nCaught KeyboardInterrupt, terminating workers")
    p.terminate()
else:
    p.close()
p.join()

# compute the average metrics
metrics = np.zeros((len(scenes), 6), dtype=float)
for i, scene in enumerate(scenes):
    scene_metrics = np.loadtxt(os.path.join(results_dir, f'{scene}.txt'))
    metrics[i] = scene_metrics
avg_metrics = np.mean(metrics, axis=0)
print(f'total      | {avg_metrics[3]:.4f} | {avg_metrics[4]:.4f} | {avg_metrics[5]:.4f}')
with open(os.path.join(results_dir, f'total.txt'), 'w') as f:
    f.write('\n'.join(f'{m:.6f}' for m in avg_metrics))
