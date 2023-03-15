# BAAM: Monocular 3D pose and shape reconstruction with bi-contextual attention module and attention-guided modeling

![sample](https://github.com/gywns6287/BAAM/blob/main/for_git/resutls.png)

## Introduction
This repo is the official Code of  BAAM: Monocular 3D pose and shape reconstruction with bi-contextual attention module and attention-guided modeling (**CVPR 2023**).

## Installation
We recommend you to use an **Anaconda** virtual environment with **Python 3.9**. 

1. Install [pytorch 1.10.1](https://pytorch.org/get-started/previous-versions/), [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), and [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```
#pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# detectron2
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
2. Install Requirements
```
pip install -r requirement.txt
```
3. Set data referring [here](https://github.com/gywns6287/BAAM/blob/main/for_git/directory.md).

## Inference
First install [pre-trained weights](https://drive.google.com/file/d/1oM-iA5Z-8AOBgX5hUCfAoLX8hcn4YBpp/view?usp=sharing) and place it in root [CODE] path. Then run the command below.
```
python main.py
```

## Train
1. To faster convergence, we use the pre-trainned COCO 2017 weights. You can downlod it from [hear](https://drive.google.com/file/d/1GZyzJLB3FTcs8C7MpZRQWw44liYPyOMD/edit).
2. Replace the third line of `configs/custom.yaml` - `best_rel_model.pth` to `CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST.pth`.
3. Run the command below.
```
python main.py -t
```

## Evaluation
1. Finish either inference process or train process.
2. move to `evaluation` folder.
3. Run the comman below.
```
python eval.py --light --test_dir ../outputs/res --gt_dir /data/apollo/val/apollo_annot --res_file test.txt
```

## Results
We achieved the state-of-the art on Apollocar3D dataset.
![table](https://github.com/gywns6287/BAAM/blob/main/for_git/resutls.png)
