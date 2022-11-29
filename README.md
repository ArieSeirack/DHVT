# DHVT

This repository contains PyTorch based codes for DHVT, including detailed models, training code and pretrained models for the NeurIPS 2022 paper:
> [Bridging the Gap Between Vision Transformers and Convolutional Neural Networks on Small Datasets](https://arxiv.org/abs/2210.05958)
> 
> _Zhiying Lu, Hongtao Xie, Chuanbin Liu, Yongdong Zhang_

## Citation
If you use this code for a paper please cite:

```
@inproceedings{
lu2022bridging,
title={Bridging the Gap Between Vision Transformers and Convolutional Neural Networks on Small Datasets},
author={Zhiying Lu and Hongtao Xie and Chuanbin Liu and Yongdong Zhang},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=bfz-jhJ8wn}
}
```

## Pretrained Model Zoo

We provide two DHVT models pretrained on ImageNet 2012.

| Method | #Params | GFLOPs| Acc@1 | Acc@5 | URL |
| --- | --- | --- | --- | --- | --- |
| DHVT-T | 6.2 | 1.4 | 77.6 | ???? | [model] (Wait-for-release) |
| DHVT-S | 23.8 | 5.1 | 82.3 | ???? | [model] (Wait-for-release) |



# Preparation
Models are trained using Python3.6 and the following packages
```
torch==1.9.0
torchvision==0.10.0
timm==0.4.12
tensorboardX==2.4
torchprofile==0.0.4
lmdb==1.2.1
pyarrow==5.0.0
einops==0.4.1
```
These packages can be installed by running `pip install -r requirements.txt`.

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```
We use the same datasets as in [DeiT](https://github.com/facebookresearch/deit). You can optionally use an LMDB dataset for ImageNet by building it using `folder2lmdb.py` and passing `--use-lmdb` to `main.py`, which may speed up data loading.

# Usage

First, clone the repository locally:
```
git clone https://github.com/youweiliang/evit.git
```
Change directory to the cloned repository by running `cd evit`, install necessary packages, and prepare the datasets.

## Training
To train `EViT/0.7-DeiT-S` on ImageNet, set the `datapath` (path to dataset) and `logdir` (logging directory) in `run_code.sh` properly and run `bash ./run_code.sh` (`--nproc_per_node` should be modified if necessary). Note that the batch size in the paper is 16x128=2048.

Set `--base_keep_rate` in `run_code.sh` to use a different keep rate, and set `--fuse_token` to configure whether to use inattentive token fusion. 

### Training/Finetuning on higher resolution images
To training on images with a (higher) resolution `h`, set `--input-size h` in `run_code.sh`.

### Multinode training
Please refer to [DeiT](https://github.com/facebookresearch/deit) for multinode training using Slurm and [submitit](https://github.com/facebookincubator/submitit).

## Finetuning
First set the `datapath`, `logdir`, and `ckpt` (the model checkpoint for finetuning) in `run_code.sh`, and then run `bash ./finetune.sh`.

## Evaluation
To evaluate a pre-trained `EViT/0.7-DeiT-S` model on ImageNet val with a single GPU run (replacing `checkpoint` with the actual file):
```
python3 main.py --model deit_small_patch16_shrink_base --fuse_token --base_keep_rate 0.7 --eval --resume checkpoint --data-path /path/to/imagenet
```
You can also pass `--dist-eval` to use multiple GPUs for evaluation. 

## Throughput
You can measure the throughput of the model by passing `--test_speed` or `--only_test_speed` to `main.py`. We also provide a script `speed_test.py` for comparing the throughput of many vision backbones (as shown in Figure 4 in the paper).

## Visualization
You can visualize the masked image (with image patches dropped/fused) by a command like this:
```
python3 main.py --model deit_small_patch16_shrink_base --fuse_token --base_keep_rate 0.7 --visualize_mask --n_visualization 64 --resume checkpoint --data-path /path/to/imagenet
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Acknowledgement
We would like to think the authors of [DeiT](https://github.com/facebookresearch/deit) and [timm](https://github.com/rwightman/pytorch-image-models), based on which this codebase was built.
