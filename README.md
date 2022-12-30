# DHVT

This repository contains PyTorch based codes for DHVT, including detailed models, training code and pretrained models for the NeurIPS 2022 paper:
> [Bridging the Gap Between Vision Transformers and Convolutional Neural Networks on Small Datasets](https://arxiv.org/abs/2210.05958)
> 
> _Zhiying Lu, Hongtao Xie, Chuanbin Liu, Yongdong Zhang_

## Updates!
Note that we fix the bug when calculating the FLOPs of the models. There are two reasons.

(1) The previous applied toolkit [fvcore](https://github.com/facebookresearch/fvcore) does not support some operations. And now we change the toolkit to [deepspeed](https://github.com/microsoft/DeepSpeed), which is more robust. (2) When calculating the FLOPs of CNNs on CIFAR-100, we made mistakes on the resolution. For example, on the previous version of the paper, the image of 32x32 in CIFAR was first pooled to 8x8 and then fed into the first stage of the CNNs. This downsampling operation greatly decreased the performance. **The correct way is to remove such downsampling**, which means the input resolution of the first stage of CNNs is kept to 32x32. And we now achieve the close results with the original CNN papers. Therefore, the correct FLOPs of CNNs is roughly 16 times as those in our original paper.

The results in the [Openreview version](https://openreview.net/forum?id=bfz-jhJ8wn) and [arxiv version](https://arxiv.org/abs/2210.05958) have been modified.

## Usage

```
git clone https://github.com/ArieSeirack/DHVT.git
```
Change directory to the cloned repository by running `cd DHVT`, install necessary packages, and prepare the datasets.

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
Download the desired datasets from the following links, and use the scripts in [VTs-Drloc](https://github.com/yhlleo/VTs-Drloc/tree/master/scripts) to pre-process the DomainNet datasets. For the CIFAR-100 dataset, we recommend using the official dataset reader code from torchvision as in the [dataset.py](https://github.com/ArieSeirack/DHVT/blob/main/datasets.py) as set the `download=True`

|Dataset|Download Link|
|:-----|:-----|
|[ImageNet](https://www.image-net.org/)|[train](http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar),[val](http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)|
|[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)|[all](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)|
|[Clipart](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt)|
|[Infograph](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt)|
|[Painting](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt)|
|[Quickdraw](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt)|
|[Real](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt)|
|[Sketch](http://ai.bu.edu/M3SDA/)|[images](http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip), [train_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt), [test_list](http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt)|

For the ImageNet-1k dataset, the directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively.

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
You can optionally use an LMDB dataset for ImageNet by building it using `folder2lmdb.py` and passing `--use-lmdb` to `main.py`, which may speed up data loading.


## Training
We provide three `run_code_[dataset].sh` file that contains the training hyperparameters.

For example, to train `DHVT-Small-CIFAR100-patch4` with 2 GPUs on single node, you can do 
```
CUDA_VISIBLE_DEVICES=0,1 bash run_code_cifar.sh
```

To train other model variants on other datasets, just follow the above operation. The `now` variable is to make the directory for output model checkpoints.


## Finetuning
Firstly, set the `ckpt` (the path to the pretrained model checkpoint) and in `finetune.sh`, and then:

```
CUDA_VISIBLE_DEVICES=0,1 bash finetune.sh
```

## Pretrained Model Zoo on ImageNet-1k

We provide two DHVT models pretrained on ImageNet 2012.

| Method | #Params | GFLOPs| Acc@1 | Acc@5 | URL |
| --- | --- | --- | --- | --- | --- |
| DHVT-T | 6.2 | 1.4 | 77.6 | 93.4 | (Wait-for-release) |
| DHVT-S | 23.8 | 5.1 | 82.3 | 96.0 | (Wait-for-release) |

## Future Work

1. Release the pretrained models on ImageNet-1k. (Coming in mid-December)

2. Recombine the code structure and split the large scripts in `vision_transformer.py` into multiple smaller ones.

3. Improve the method to _DHVTv2_, which is a hierarchical structure and with lower computational costs and higher performance

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Acknowledgement
We would like to thank the authors of [DeiT](https://github.com/facebookresearch/deit), [timm](https://github.com/rwightman/pytorch-image-models), [VTs-Drloc](https://github.com/yhlleo/VTs-Drloc), [XCiT](https://github.com/facebookresearch/xcit), [CeiT](https://github.com/coeusguo/ceit) and mainly [EViT](https://github.com/youweiliang/evit), based on which this codebase was built.

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
