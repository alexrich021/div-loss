import os
from pathlib import Path
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, default=os.path.join(os.path.expanduser('~'), 'scannetpp'),
                    help='Path to raw ScanNet++ dataset')
parser.add_argument('--split', default='val', choices=['train', 'val', 'both'],
                    help='Which split to pre-process')
parser.add_argument('--data_type', default='dslr', choices=['dslr', ],
                    help='Currently, only pre-processing of the DSLR images is supported')
parser.add_argument('--out_dir', type=Path, default=os.path.join(os.path.expanduser('~'), 'scannetpp_mvs'),
                    help='Where to store pre-processed dataset')
args = parser.parse_args()
args.out_dir.mkdir(exist_ok=True)

scenes = []
if args.split == 'train' or args.split == 'both':
    with open(args.data_dir / 'splits' / 'nvs_sem_train.txt', 'r') as f:
        scenes.extend([l.rstrip() for l in f.readlines()])
if args.split == 'val' or args.split == 'both':
    with open(args.data_dir / 'splits' / 'nvs_sem_val.txt', 'r') as f:
        scenes.extend([l.rstrip() for l in f.readlines()])

for i, scene in enumerate(scenes):
    print(f'{i+1:03} / {len(scenes)} | {scene}')
    scene_dir = args.data_dir / 'data' / scene / args.data_type
    colmap_dir = scene_dir / 'colmap'
    out_dir = args.out_dir / scene
    out_dir.mkdir(exist_ok=True)

    # first, run image undistortion
    images_dir = scene_dir / 'resized_images'
    tmp_dir = scene_dir / 'tmp'
    cmd = ' '.join([
        'colmap image_undistorter',
        f'--image_path {images_dir}',
        f'--input_path {colmap_dir}',
        f'--output_path {tmp_dir}'
    ])
    print(cmd)
    os.system(cmd)

    # next, run view selection
    undistorted_images_dir = tmp_dir / 'images'
    undistorted_colmap_dir = tmp_dir / 'sparse'
    cmd = ' '.join([
        'python colmap2mvsnet.py',
        f'--image_dir {undistorted_images_dir}',
        f'--model_dir {undistorted_colmap_dir}',
        f'--out_dir {out_dir}',
        '--model_ext .bin',
        '--max_d 192'
    ])
    print(cmd)
    os.system(cmd)

    # delete tmp dir
    shutil.rmtree(str(tmp_dir))

    # finally, copy GT mesh
    mesh_src_file = args.data_dir / 'data' / scene / 'scans' / 'mesh_aligned_0.05.ply'
    mesh_dst_file = out_dir / 'mesh_aligned_0.05.ply'
    shutil.copyfile(mesh_src_file, mesh_dst_file)
