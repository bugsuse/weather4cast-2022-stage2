#!/bin/sh

model=unet3d
mode=heldout
loss=ioudice
config=unet3d_ioudice_shuffle_crop_stage2.yaml
name=${model}_${loss}_${mode}
ckpt=weights/unet3d.ioudice.epoch47.pth
pack=core.${model}.${loss}.shuffle.crop2.epoch47
nomask=True

time python train.py ${model} \
    --mode ${mode} \
    --gpus 1 \
    --config_path ${config}\
    --name ${name}\
    -l ${loss} \
    -c ${ckpt}\
    -savedir submission.heldout.core \
    -y 2019,2020 \
    -r boxi_0015,boxi_0034,boxi_0076,roxi_0004,roxi_0005,roxi_0006,roxi_0007 \
    -nomask ${nomask} \
    -p ${pack}
