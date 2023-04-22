<div align="center">
  <h1>Self-Taught Metric Learning without Labels <br> (CVPR 2022)</h1>
</div>

<div align="center">
  <h3><a href=https://cvlab.postech.ac.kr/~sungyeon>Sungyeon Kim</a>, <a href=https://github.com/kdwonn>Dongwon Kim</a>, <a href=https://cvlab.postech.ac.kr/~mcho>Minsu Cho</a>, <a href=https://suhakwak.github.io/>Suha Kwak</a></h3>
</div>

<div align="center">
  <h4> <a href=http://arxiv.org/abs/2205.01903>[paper]</a>, <a href=http://cvlab.postech.ac.kr/research/STML>[project hompage]</a></h4>
</div>

<br>

Official PyTorch implementation of CVPR 2022 paper **Self-Taught Metric Learning without Labels (STML)**. 
A standard embedding network trained with STML achieves SOTA performance on unsupervised metric learning and sometimes even beats supervised learning models.
This repository provides source code of unsupervised metric learning experiments on three datasets (CUB-200-2011, Cars-196, Stanford Online Products).

## New Update
- [23.04.22] Code for training and testing an embedding network with SSL Methods (MoCo, BYOL, MeanShift).
- [23.04.22] Fix bug about "same device as the indexed tensor" in Evaluation.

## Overview

### Self-Taught Metric Learning
1. Contextualized semantic similarity between a pair of data is estimated on the embedding space of the teacher network. 
2. The semantic similarity is then used as a pseudo label, and the student network is optimized by relaxed contrastive loss with KL divergence.
3. The teacher network is updated by an exponential moving average of the student.
  
<p align="center"><img src="misc/stml_overview.png" alt="graph" width="90%"></p>

### Experimental Restuls
- Our model with 128 embedding dimensions outperforms all previous arts using higher embedding dimensions and sometimes surpasses supervised learning methods. 

<p align="center"><img src="misc/stml_recall.png" alt="graph" width="70%"></p>

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- [AdamP](https://github.com/clovaai/AdamP)

## Datasets

1. Download four public benchmarks for deep metric learning
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))

2. Extract the tgz or zip file into `./data/` (Exceptionally, for Cars-196, put the files in a `./data/cars196`)

## Training Embedding Network

### CUB-200-2011 (Unsupervised)

- Train an embedding network with GoogLeNet (d=512) using STML

```bash
python3 code/main.py --gpu-id 0 \
                        --model googlenet \
                        --embedding_size 512 \
                        --optimizer adamp \
                        --lr 1e-4 \
                        --dataset cub \
                        --view 2 \
                        --sigma 3 \
                        --delta 1 \
                        --num_neighbors 5
```

- Train an embedding network with BN-Inception (d=512) using STML

```bash
python3 code/main.py --gpu-id 0 \
                        --model bn_inception \
                        --embedding_size 512 \
                        --optimizer adamp \
                        --lr 1e-4 \
                        --dataset cub \
                        --view 2 \
                        --sigma 3 \
                        --delta 1 \
                        --num_neighbors 5 \
                        --bn-freeze 1
```

### Cars-196 (Unsupervised)

- Train an embedding network with GoogLeNet (d= 512) using STML

```bash
python3 code/main.py --gpu-id 0 \
                        --model googlenet \
                        --embedding_size 512 \
                        --optimizer adamp \
                        --lr 1e-4 \
                        --dataset cars \
                        --view 2 \
                        --sigma 3 \
                        --delta 1 \
                        --num_neighbors 5
```

- Train an embedding network with BN-Inception (d=512) using STML 

```bash
python3 code/main.py --gpu-id 0 \
                        --model bn_inception \
                        --embedding_size 512 \
                        --optimizer adamp \
                        --lr 1e-4 \
                        --dataset cars \
                        --view 2 \
                        --sigma 3 \
                        --delta 1 \
                        --num_neighbors 5 \
                        --bn-freeze 1
```

### Stanford Online Products (Unsupervised)

- Train an embedding network with GoogLeNet (d= 512) using STML

```bash
python3 code/main.py --gpu-id 0 \
                        --model googlenet \
                        --embedding_size 512 \
                        --optimizer adamp \
                        --lr 1e-4 \
                        --dataset SOP \
                        --view 2 \
                        --sigma 3 \
                        --delta 0.9 \
                        --num_neighbors 2 \
                        --momentum 0.9 \
                        --weight-decay 1e-2 \
                        --emb-lr 1e-2
```

- Train an embedding network with BN-Inception (d=512) using STML 

```bash
python3 code/main.py --gpu-id 0 \
                        --model bn_inception \
                        --embedding_size 512 \
                        --optimizer adamp \
                        --lr 1e-4 \
                        --dataset SOP \
                        --view 2 \
                        --sigma 3 \
                        --delta 0.9 \
                        --num_neighbors 2 \
                        --momentum 0.9 \
                        --weight-decay 1e-2 \
                        --emb-lr 1e-2 \
                        --bn_freeze 1
```

### Stanford Online Products (Unsupervised & From Scratch)

- Train an embedding network with ResNet18 (d=128) using STML 

```bash
python3 code/main.py --gpu-id 0 \
                        --model resnet18 \
                        --embedding_size 128 \
                        --optimizer adamp \
                        --lr 5e-4 \
                        --dataset SOP \
                        --view 2 \
                        --sigma 3 \
                        --delta 0.9 \
                        --num_neighbors 2 \
                        --momentum 0.9 \
                        --pretrained false \
                        --weight-decay 1e-2 \
                        --batch-size 120 \
                        --epoch 180 \
                        --fix_lr true
```

## Training Embedding Network using SSL Method

### Example using MoCo

- Train an embedding network with GoogLeNet (d=512) using MoCo

```bash
python3 code/main_SSL.py --gpu-id 0 \
                        --model googlenet \
                        --embedding_size 512 \
                        --optimizer adamp \
                        --lr 1e-4 \
                        --dataset cub \
                        --view 2 \
                        --method moco \
                        --memory-size 9600
```

### Example using MeanShift

- Train an embedding network with GoogLeNet (d=512) using MeanShift
- Note that BYOL is same with MeanShift using topk 1

```bash
python3 code/main_SSL.py --gpu-id 0 \
                        --model googlenet \
                        --embedding_size 512 \
                        --optimizer adamp \
                        --lr 1e-4 \
                        --dataset cub \
                        --view 2 \
                        --method meanshift \
                        --memory-size 9600 \
                        --topk 5 
```

## Acknowledgements

Our source code is modified and adapted on these great repositories:

- [Embedding Transfer with Label Relaxation for Improved Metric Learning](https://github.com/tjddus9597/LabelRelaxation-CVPR21)
- [Proxy Anchor Loss for Deep Metric Learning](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)
- [No Fuss Distance Metric Learning using Proxies](https://github.com/dichotomies/proxy-nca)
- [PyTorch Metric learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://github.com/facebookresearch/moco)
- [Mean Shift for Self-Supervised Learning](https://github.com/UMBCvision/MSF)


## Citation

If you use this method or this code in your research, please cite as:

    @inproceedings{kim2022self,
      title={Self-Taught Metric Learning without Labels},
      author={Kim, Sungyeon and Kim, Dongwon and Cho, Minsu and Kwak, Suha},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      year={2022}
    }