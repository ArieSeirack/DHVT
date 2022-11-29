# DHVT

This repository contains PyTorch based codes for DHVT, including detailed models, training code and pretrained models for the NeurIPS 2022 paper:
> [Bridging the Gap Between Vision Transformers and Convolutional Neural Networks on Small Datasets](https://arxiv.org/abs/2210.05958)
> 
> _Zhiying Lu, Hongtao Xie, Chuanbin Liu, Yongdong Zhang_


## Preparation
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
Download the desired datasets from the following link.

|Dataset|Download Link|
|:-----|:-----|
|[ImageNet](https://www.image-net.org/)|[train](http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar),[val](http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)|
|[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)|[all](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)|
|[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)|[all](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)|
|[Clipart](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt)|
|[Infograph](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt)|
|[Painting](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt)|
|[Quickdraw](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt)|
|[Real](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt)|
|[Sketch](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt)|

The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:
Except for CIFAR-100, other datasets should be arranged as the following structure:
```
[dataset_name]
  |__train
  |    |__class1
  |    |    |__www.jpg
  |    |    |__...
  |    |__class2
  |    |    |__xxx.jpg
  |    |    |__...
  |    |__...
  |__val
       |__class1
       |    |__yyy.jpg
       |    |__...
       |__class2
       |    |__zzz.jpg
       |    |__...
       |__...
```
We use the same datasets as in [DeiT](https://github.com/facebookresearch/deit). You can optionally use an LMDB dataset for ImageNet by building it using `folder2lmdb.py` and passing `--use-lmdb` to `main.py`, which may speed up data loading.

## Usage

```
git clone https://github.com/ArieSeirack/DHVT.git
```
Change directory to the cloned repository by running `cd DHVT`, install necessary packages, and prepare the datasets.

## Training
To train `EViT/0.7-DeiT-S` on ImageNet, set the `datapath` (path to dataset) and `logdir` (logging directory) in `run_code.sh` properly and run `bash ./run_code.sh` (`--nproc_per_node` should be modified if necessary). Note that the batch size in the paper is 16x128=2048.

Set `--base_keep_rate` in `run_code.sh` to use a different keep rate, and set `--fuse_token` to configure whether to use inattentive token fusion. 


## Finetuning
Firstly, set the `ckpt` (the path to the pretrained model checkpoint) and in `run_code.sh`, and then:
run `bash ./finetune.sh`.

## Pretrained Model Zoo on ImageNet-1k

We provide two DHVT models pretrained on ImageNet 2012.

| Method | #Params | GFLOPs| Acc@1 | Acc@5 | URL |
| --- | --- | --- | --- | --- | --- |
| DHVT-T | 6.2 | 1.4 | 77.6 | ???? | [model] (Wait-for-release) |
| DHVT-S | 23.8 | 5.1 | 82.3 | ???? | [model] (Wait-for-release) |

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Acknowledgement
We would like to think the authors of [DeiT](https://github.com/facebookresearch/deit) and [timm](https://github.com/rwightman/pytorch-image-models), based on which this codebase was built.

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
