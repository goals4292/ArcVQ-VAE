# ArcVQ-VAE

## Reconstruction Result
![teaser](assets/rec.png)

original images (top), baseline VQGAN (middle), our proposed ArcVQ-VAE (bottom)

## Generation Result
![teaser](assets/gen.png)

# VQ-VAE ver

## Requirements
A suitable [conda](https://conda.io/) environment named `arcvq-vae` can be created
and activated with:

```
cd ArcVQ-VAE
conda env create -f environment.yaml
conda activate arcvq-vae
```

## Model Training

Training can be started by running:

```
python main.py \
--data_folder ./mnist \
--dataset mnist \
--output_folder ./output \
--exp_name arcvqvae_mnist \
--batch_size 1024 \
--device cuda \
--num_epochs 500 \
--num_embedding 512 \
--embedding_dim 64
```

We also support training on CIFAR-10.  
To train on CIFAR-10, replace `mnist` with `cifar10` in the dataset-related arguments:

```
python main.py \
--data_folder ./cifar10 \
--dataset cifar10 \
--output_folder ./output \
--exp_name arcvqvae_cifar10 \
--batch_size 1024 \
--device cuda \
--num_epochs 500 \
--num_embedding 512 \
--embedding_dim 64
```

We also provide a pretrained checkpoint for the VQ-VAE version.
Download `best.pt` from [here](https://drive.google.com/drive/folders/1FoHlXrrFgEVvegFNkk-AjzEofZ03IR9k?usp=sharing) and place it under `./output/arcvqvae_mnist/best.pt` or `./output/arcvqvae_cifar10/best.pt`.

## Model Testing

```
python test.py \
--data_folder ./mnist \
--dataset mnist \
--output_folder ./output \
--model_name arcvqvae_mnist/best.pt \
--batch_size 16 \
--device cuda \
--num_embedding 512 \
--embedding_dim 64 \
```

The default results will be stored under the ```<output_folder>/results/<model_name>``` folder, in which:
- ```original/```: shows original images
- ```rec/```: shows reconstruction images


## Model Evaluation

```
python evaluation.py \
--gt_path ./output/results/arcvqvae_mnist/best.pt/original/ \
--g_path ./output/results/arcvqvae_mnist/best.pt/rec/
```




# VQGAN ver

## Requirements
A suitable [conda](https://conda.io/) environment named `taming` can be created
and activated with:

```
cd taming-transformers
conda env create -f environment.yaml
conda activate taming
```


## Data Preparation

### ImageNet
The code will try to download (through [Academic
Torrents](http://academictorrents.com/)) and prepare ImageNet the first time it
is used. However, since ImageNet is quite large, this requires a lot of disk
space and time. If you already have ImageNet on your disk, you can speed things
up by putting the data into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` (which defaults to
`~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`), where `{split}` is one
of `train`/`validation`. It should have the following structure:

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── ...
├── ...
```

If you haven't extracted the data, you can also place
`ILSVRC2012_img_train.tar`/`ILSVRC2012_img_val.tar` (or symlinks to them) into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/` /
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_validation/`, which will then be
extracted into above structure without downloading it again.  Note that this
will only happen if neither a folder
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` nor a file
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/.ready` exist. Remove them
if you want to force running the dataset preparation again.


## Model Training

Train a VQGAN with
```
python main.py --base configs/imagenet_vqgan.yaml -t True --gpus 0,
```

The VQGAN version uses the ImageNet configuration at
`configs/imagenet_vqgan.yaml`. ArcVQ-VAE-specific hyperparameters can be
controlled directly from this YAML file:

```
model:
  params:
    bbnr_alpha: 0.00005

    lossconfig:
      params:
        arc_gamma: 1.0
        arc_gamma_decay: 1.0e-4
        arc_s: 10.0
        arc_m: 0.1
        arc_top_k: 3
```

The ArcLoss parameters control the angular-margin regularization applied during
the VQGAN autoencoder update:

- `bbnr_alpha`: growth rate of the codebook norm bound in BBNR.
- `arc_gamma`: initial weight of the ArcLoss.
- `arc_gamma_decay`: decay rate of the ArcLoss weight during training.
- `arc_s`: scale factor for angular similarity in ArcLoss.
- `arc_m`: additive angular margin used in ArcLoss.
- `arc_top_k`: number of nearest latent tokens used as positives for each codebook vector.


# Acknowledgements

This repository is built upon the official implementation of [CVQ-VAE](https://github.com/lyndonzheng/CVQ-VAE), [taming-transformers](https://github.com/CompVis/taming-transformers).

We gratefully acknowledge the authors for making their implementation publicly available.
