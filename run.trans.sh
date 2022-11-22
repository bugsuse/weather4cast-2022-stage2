#!/bin/sh

model=unet3d
mode=heldout
loss=ioudice
config=unet3d_ioudice_shuffle_crop_stage2.yaml
name=${model}_${loss}_${mode}
ckpt=weights/unet3d.ioudice.epoch47.pth
pack=trans.${model}.${loss}.shuffle.crop2.epoch47
nomask=True


time python trainv3.py ${model} \
    --mode ${mode} \
    --gpus 0 \
    --config_path ${config}\
    --name ${name}\
    -l ${loss} \
    -c ${ckpt}\
    -savedir submission.heldout.trans \
    --years 2019,2020 \
    --nomask ${nomask} \
    --regions roxi_0008,roxi_0009,roxi_0010

time python trainv3.py ${model} \
    --mode ${mode} \
    --gpus 0 \
    --config_path ${config}\
    --name ${name}\
    -l ${loss} \
    -c ${ckpt}\
    -savedir submission.heldout.trans \
    -y 2021 \
    --nomask ${nomask} \
    -r boxi_0015,boxi_0034,boxi_0076,roxi_0004,roxi_0005,roxi_0006,roxi_0007,roxi_0008,roxi_0009,roxi_0010 \
    -p ${pack}
