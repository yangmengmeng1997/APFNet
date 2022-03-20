Deep Adaptive Fusion Network for High Performance RGBT Tracking
This project is created base on
-- RT-MDNet: Real-Time Multi-Domain Convolutional Neural Network Tracker Created by Ilchae Jung, Jeany Son, Mooyeol Baek, and Bohyung Han

Introduction
RT-MDNet is the real-time extension of MDNet and is the state-of-the-art real-time tracker. Detailed description of the system is provided by our project page and paper

Citation
If you're using this code in a publication, please cite our paper.

@InProceedings{rtmdnet,
author = {Jung, Ilchae and Son, Jeany and Baek, Mooyeol and Han, Bohyung},
title = {Real-Time MDNet},
booktitle = {European Conference on Computer Vision (ECCV)},
month = {Sept},
year = {2018}
}
System Requirements
This code is tested on 64 bit Linux (Ubuntu 16.04 LTS).

Prerequisites 0. PyTorch (>= 0.2.1) 0. For GPU support, a GPU (~2GB memory for test) and CUDA toolkit. 0. Training Dataset (ImageNet-Vid) if needed.

Online Tracking
Pretrained Model and results If you only run the tracker, you can use the pretrained model: RT-MDNet-ImageNet-pretrained. Also, results from pretrained model are provided in here.

Demo 0. Run 'Run.py'.

Learning RT-MDNet
Preparing Datasets 0. If you download ImageNet-Vid dataset, you run 'modules/prepro_data_imagenet.py' to parse meta-data from dataset. After that, 'imagenet_refine.pkl' is generized. 0. type the path of 'imagenet_refine.pkl' in 'train_mrcnn.py'

Demo 0. Run 'train_mrcnn.py' after hyper-parameter tuning suitable to the capacity of your system.
