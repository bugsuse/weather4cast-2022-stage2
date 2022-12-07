This code was used for the entry by the team "meteoai" for the Weather4cast 2022 NeurIPS Competition on Stage2. Below, you can find the instructions for generating predictions, evaluating pre-trained models and training new models.

This repo is the official implementation of "[Super-resolution Probabilistic Rain Prediction from Satellite Data Using 3D U-Nets and EarthFormers](https://zenodo.org/record/7405710)".

## Installation

To use the code, you need to:
1. Clone the repository.
1. Setup a conda environment. You can find an environment verified to work in the `environment.yaml` file. However, you might have to adapt it to your own CUDA installation.
1. Fetch the data you want from the competition website. Follow the instructions [here](https://github.com/iarai/weather4cast-2022#Get the data). The data should be in the `data` directory following the structure specified [here](https://github.com/iarai/weather4cast-2022#Starter kit).
1. (Optional) If you want to use the pre-trained models, download them from https://doi.org/10.5281/zenodo.7339193 . Place the `.pth` files in the `weights` directory.


## Running the code

There you can either launch the `train.py` script with instructions provided below.

### Train a model
```bash
## Training the Unet3d model
python train.py unet3d --gpus 0 --config_path unet3d_ioudice_shuffle_crop_stage2.yaml --name unet3d_ioudice_shuffle_crop_stage2 -l ioudice -sigmoid False

## Training the EarthFormer model
python train.py earthformer --gpus 0 --config_path earthformer_ioudice_shuffle_crop_stage2.yaml --name earthformer_ioudice_shuffle_Crop_stage2 -l ioudice -sigmoid False
```

A GPU is basically mandatory. The default batch size is set to 64(12) for the [Unet3d](https://github.com/iarai/weather4cast-2022)([EarthFormer](https://github.com/amazon-science/earth-forecasting-transformer)) used in the study but you may have to reduce it if you don't have a lot of GPU memory.

### Evaluate a single pre-trained model
```bash
python train.py unet3d --gpus 0 --mode val --config_path unet3d_ioudice_shuffle_crop_stage2.yaml --ckpt weights/unet3d.ioudice.epoch3.pth -l ioudice

## OR

python train.py earthformer --gpus 0 --mode val --config_path unet3d_ioudice_shuffle_Crop_stage2.yaml --ckpt weights/earthformer.ioudice.epoch8.pth -l ioudice
```

A GPU is recommended for this although in principle it can be done on a CPU.


### Reproduce predictions
Run:
```bash
sh run.core.sh ## Core Leaderboard

sh run.trans.sh ## Transfer Leaderboard
```

### Model ensemble
Run:
```bash
cd ensemble

## Core leaderboard
python ensemble.core.py pack.prediction --models unet3d.ioudice.ep47 unet3d.ioudicefocal.ep40 earthformerv1.ioudice.ep3 earthformerv2.ioudice.ep8 --weights 1 1 1 1

## Transfer Leaderboard
python ensemble.trans.py pack.prediction --models unet3d.ioudice.ep47 unet3d.ioudicefocal.ep40 earthformerv1.ioudice.ep3 earthformerv2.ioudice.ep8 --weights 1 1 1 1
```

> Note: You should run the inference scripts (run.core.sh or run.trans.sh) above before ensemble multi-model prediction.


## Crediets
Third-party libraries/repo:
* [PyTorch](https://pytorch.org/)
* [PyTorch Lightning](https://www.pytorchlightning.ai/)
* [weather4cast-2022](https://github.com/iarai/weather4cast-2022)
* [earth-forecasting-transformer](https://github.com/amazon-science/earth-forecasting-transformer)

## License
This project is licensed under the Apache-2.0 License.

