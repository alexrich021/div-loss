<div align="center">
<h1 align="center">Smoothness, Synthesis, and Sampling:<br>Re-thinking Unsupervised Multi-View Stereo with DIV Loss</h1>
<p>
    <a href="https://alexrich021.github.io">Alex Rich</a>, <a href="https://noahstier.github.io">Noah Stier</a>, <a href="https://web.ece.ucsb.edu/~psen/">Pradeep Sen</a>, <a href="https://sites.cs.ucsb.edu/~holl">Tobias H&ouml;llerer</a>
</p>
<p>University of California, Santa Barbara</p>
<p align="center">
    <strong>ECCV 2024</strong>
</p>
<p>
    <a href="https://alexrich021.github.io/div-loss/">Project Page</a> | <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08045.pdf">Paper PDF</a>
</p>
</div>

## Installation

The code is tested with Python == 3.9, PyTorch == 1.13.0 and CUDA == 11.6 on NVIDIA GeForce RTX 3090.
We recommend using [anaconda](https://www.anaconda.com/) to manage dependencies.
You may need to change the torch and cuda version according to your computer.
For troubleshooting the PyTorch3D install, see the [install documentation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
```
conda create -n unsupmvs python=3.9
conda activate unsupmvs
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# pytorch3d install (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

# remaining dependencies
pip install -r requirements.txt
```

## Datasets

All provided scripts assume the dataset folders are placed in the home folder.

### DTU
**Training**

Download the [DTU dataset](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) pre-processed by [MVSNet](https://github.com/YoYo000/MVSNet) and extract the archive. 
See [MVSNet](https://github.com/YoYo000/MVSNet) for file formats.
Download the original resolution [depth maps](https://drive.google.com/open?id=1LVy8tsWajG3uPTCYPSxDvVXFCdIYXaS-) provided by [YaoYao](https://github.com/YoYo000/MVSNet/issues/106). Extract it and rename the folder to `Depths_raw`.
Merge the folders together and you should get a dataset folder like below:

```
dtu
├── Cameras
├── Depths
├── Depths_raw
└── Rectified
```

**Testing**

Download the [DTU testing dataset](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) pre-processed by [MVSNet](https://github.com/YoYo000/MVSNet) and extract the archive.
See [MVSNet](https://github.com/YoYo000/MVSNet) for file formats.
We rename the dtu test set to `dtu_eval` so it can be placed in the same parent folder as the training set.

```
dtu_eval
├── scan1
├── scan4
├── scan9
...
├── scan114
└── scan118
```

### Tanks and Temples (testing only)

Download the [Tanks and Temples testing set](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) pre-processed by [UniMVSNet](https://github.com/prstrive/UniMVSNet) and unzip.
We use the standard short range camera parameters for the intermediate subset and Temple (see [this](https://github.com/YoYo000/MVSNet/issues/14) GitHub issue).
The folder structure should look like this:

```
tanksandtemples
├── advanced
│   ├── Auditorium
│   │   ├── cams
│   │   ├── images
│   │   ├── Auditorium.log
│   │   └── pair.txt
│   ├── Ballroom
│   └── ...
└── intermediate
    ├── Family
    │   ├── cams
    │   ├── images
    │   ├── Family.log
    │   └── pair.txt
    ├── Francis
    └── ...
```

### ScanNet++ (testing only)

Download the ScanNet++ dataset from the [official website](https://kaldir.vc.in.tum.de/scannetpp/) and place in your home folder.
Run `data_preprocess/preprocess_scannetpp.py` to pre-process the ScanNet++ validation set for use with MVS.
The pre-processed dataset should now be stored at `$HOME/scannetpp_mvs`.
We also include our evaluation script, `run_scannetpp_eval.py`.

## Re-producing Results

We include code for re-producing the **DIV-MVS** results.
Scripts for re-producing all **DIV-MVS** reconstructions can be found in the `scripts` folder and are named appropriately.
All scripts assume the datasets have been downloaded and placed in your home directory.
If you wish to place them elsewhere, modify the `--testpath` argument in the script.
The `scripts` folder also contains a training script for re-training **DIV-MVS**. These are set up for multi-GPU training.
If changing the dataset location to somewhere other than the home folder, set the `DATASET_DIR` variable in the script.

The default depth-smoothness weight and clip value for **DIV-MVS** are `0.36` and `2.0` respectively, which is different from values reported in the paper. To reproduce the **exact** paper results, **this is correct**; however, DIV-MVS achieves comparable results with paper hyperparameters. There was a small discrepancy between **DIV-MVS** and **DIV-RC/CL** depth-smoothness implementations, resulting in the hyperparameters reported in the paper being off for **DIV-MVS.** For **DIV-RC/CL**, the reported hyperparameters of `0.18` and `4.0` are correct.

## Acknowledgments

We made use of the following code-bases:

* [RC-MVSNet](https://github.com/Boese0601/RC-MVSNet)
* [MVSNet](https://github.com/YoYo000/MVSNet)
* [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet)
* [KD-MVS](https://github.com/megvii-research/KD-MVS)
* [UniMVSNet](https://github.com/prstrive/UniMVSNet)
