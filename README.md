
# Self-Taught Metric Learning without Labels

Supplementary material provides source code of experiments on three datasets (CUB-200-2011, Cars-196 and Stanford Online Products) and pretrained models.

Please download checkpoint files of STML and pre-trained model (SwAV, ImageNet) through this [Link](https://drive.google.com/file/d/1Kh2kToqhZG9GHlkHGKPlKo844evnwMN1/view?usp=sharing). <br>
(There are separate folders for each dataset, and the checkpoint file is in a folder with its settings as a name.)

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb

## Datasets

1. Download four public benchmarks for deep metric learning
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))

2. Extract the tgz or zip file into `./data/` (Exceptionally, for Cars-196, put the files in a `./data/cars196`)

## Training Target Embedding Network

### CUB-200-2011 (Unsupervised)

- Train a target embedding network with GoogLeNet (d=512) using STML

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
                        --IPC 5
```

- Train a target embedding network with BN-Inception (d=512) using STML

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
                        --IPC 5 \
                        --bn-freeze 1
```

### Cars-196 (Unsupervised)

- Train a target embedding network with GoogLeNet (d= 512) using STML

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
                        --IPC 5
```

- Train a target embedding network with BN-Inception (d=512) using STML 

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
                        --IPC 5 \
                        --bn-freeze 1
```

### Stanford Online Products (Unsupervised)

- Train a target embedding network with GoogLeNet (d= 512) using STML

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
                        --IPC 2 \
                        --momentum 0.9
```

- Train a target embedding network with BN-Inception (d=512) using STML 

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
                        --IPC 2 \
                        --momentum 0.9 \
                        --bn_freeze 1
```

### Stanford Online Products (Unsupervised & From Scratch)

- Train a target embedding network with ResNet18 (d=128) using STML 

```bash
python3 code/main.py --gpu-id 0 \
                        --model resnet18 \
                        --embedding_size 128 \
                        --optimizer adamp \
                        --lr 5e-4 \
                        -- fix_lr true \
                        --dataset SOP \
                        --view 2 \
                        --sigma 3 \
                        --delta 0.9 \
                        --IPC 2 \
                        --momentum 0.9 \
                        --pretrained false \
                        --weight-decay 1e-2 \
                        --batch-size 120 \
                        --epoch 180
```