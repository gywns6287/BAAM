# BAAM: Monocular 3D pose and shape reconstruction with bi-contextual attention module and attention-guided modeling

![sample](https://github.com/gywns6287/BAAM/blob/main/for_git/resutls.png)

## Introduction
This repo is the official Code of  BAAM: Monocular 3D pose and shape reconstruction with bi-contextual attention module and attention-guided modeling (**CVPR 2023**). [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_BAAM_Monocular_3D_Pose_and_Shape_Reconstruction_With_Bi-Contextual_Attention_CVPR_2023_paper.html)

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
1. To stable model convergence, we first trained the 2D modules (Box, Keypoint) based on the pre-trainned [COCO 2017 weights](https://drive.google.com/file/d/1GZyzJLB3FTcs8C7MpZRQWw44liYPyOMD/edit). You can downlod pre-trained 2D module weights (res2net_bifpn.pth) in [here](https://drive.google.com/file/d/1aX_-SfHtXAdE-frgrbrlQYuWddhwX3V3/view?usp=drive_link).
2. Replace the third line of `configs/custom.yaml` - `best_rel_model.pth` to `res2net_bifpn.pth`.
3. Run the command below.
```
python main.py -t
```

## Evaluation
1. Finish either inference process or train process.
2. Move to `evaluation` folder.
3. Run the command below.
```
python eval.py --light --test_dir ../outputs/res --gt_dir ../data/apollo/val/apollo_annot --res_file test_results.txt
```
4. You can show A3DP results in `test_results.txt`.

## Visualization
1. Install open3D python library
```
pip install open3d==0.14.1
```
**note** must use version 0.14.1 

2. Move to 'vis' folder.
3. Run the command below.
```
python vis_apollo.py --output [path where the results are saved] --file [file name to vis] --save [path to save vis results]
python vis_apollo.py --output ../outputs --file 171206_081122658_Camera_5 --save vis_results #example
```
4. You can see a manual to handle open3D UI [here](http://www.open3d.org/docs/latest/tutorial/visualization/visualization.html).
5. You can see the vis results at [save] path.
  - [file].image_plane.png : vis results rendered on an image plane.
  - [file].3d.png: vis results of your own rendering with open3d UI.

## Results
We achieved the state-of-the art on Apollocar3D dataset.
![table](https://github.com/gywns6287/BAAM/blob/main/for_git/table.png)

## License

A MIT license is used for this repository. Note that the used dataset (ApolloCar3D) is subject to their respective licenses and may not grant commercial use.
