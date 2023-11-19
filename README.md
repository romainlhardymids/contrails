# RSNA Abdominal Trauma Detection Challenge

# Introduction
This repository contains the training and inference code for my solution to the [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming), which placed 76th on the public leaderboard and 65th on the private leaderboard. 
My solution is an ensemble of 6 models. The first model is a binary classification model that seeks to filter out satellite images that contain no contrails. The remaining five models are segmentators that predict probability maps of potential contrails for the remaining images. 
4 of the segmentation models are trained on 2D images, and one model is trained on 3D images. The full submission notebook is included in the `notebooks` folder for reference.

The commands below can be used to execute the training scripts.

```
# Classification
python3 src/scripts/classification/train.py --config=configs/classification/tf_efficientnetv2_s_512.yaml

# 2D segmentation
python3 src/scripts/2d/train.py --config=configs/segmentation/2d_tf_efficientnetv2_l_512.yaml

# 3D segmentation
python3 src/scripts/3d/train.py --config=configs/segmentation/3d_tf_efficientnetv2_m_512.yaml
```
