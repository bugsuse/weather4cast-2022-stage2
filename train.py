# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
#
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


import os
import ast
import warnings
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from models.trainer import ModelTrainer
from utils.data_utils import load_config
from utils.data_utils import get_cuda_memory_usage
from utils.data_utils import tensor_to_submission_file
from utils.w4c_dataloader import RainData

warnings.filterwarnings("ignore")
seed_everything(100)


class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """
    def __init__(self, params, training_params, mode):
        super().__init__()
        self.params = params
        self.training_params = training_params

        if mode in ['train']:
            print("Loading TRAINING/VALIDATION dataset -- as test")
            self.train_ds = RainData('training', **self.params)
            self.val_ds = RainData('validation', **self.params)
        if mode in ['val']:
            print("Loading VALIDATION dataset -- as test")
            self.val_ds = RainData('validation', **self.params)
        if mode in ['predict']:
            print("Loading PREDICTION/TEST dataset -- as test")
            self.test_ds = RainData('test', **self.params)
        if mode in ['heldout']:
            print("Loading HELD-OUT dataset -- as test")
            self.test_ds = RainData('heldout', **self.params)

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(dataset,
                        batch_size=self.training_params['batch_size'],
                        num_workers=self.training_params['n_workers'],
                        shuffle=shuffle, pin_memory=pin, prefetch_factor=2,
                        persistent_workers=False)
        return dl

    def train_dataloader(self):
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)

    def val_dataloader(self):
        return self.__load_dataloader(self.val_ds, shuffle=False, pin=True)

    def test_dataloader(self):
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)


def load_model(model_name, params, checkpoint_path=''):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    p = {**params['experiment'], **params['dataset'], **params['train']}

    if checkpoint_path == '':
        print('-> Modelling from scratch!  (no checkpoint loaded)')
        model = ModelTrainer(model_name, params['model'], p)
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        if '.pth' in checkpoint_path:
            checkpoint_path = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))
            model = ModelTrainer(model_name, params['model'], p)
            model.load_state_dict(checkpoint_path)
        else:
            model = ModelTrainer.load_from_checkpoint(checkpoint_path, model_name=model_name,
                                                      model_params=params['model'], params=p,
                                                      strict=False)
    return model


def get_trainer(gpus, params, options):
    """ get the trainer, modify here its options:
        - save_top_k
     """
    max_epochs=params['train']['max_epochs'];
    print("Trainig for", max_epochs, "epochs");
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', save_top_k=-1, save_last=True,
                                          filename='{epoch:02d}-{val_loss_epoch:.6f}')

    parallel_training = None
    ddpplugin = None
    if gpus[0] == -1:
        gpus = None
    elif len(gpus) > 1:
        parallel_training = 'ddp'
##        ddpplugin = DDPPlugin(find_unused_parameters=True)

    print(f"====== process started on the following GPUs: {gpus} ======")
    date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
    version = params['experiment']['name']
    version = version + '_' + date_time

    #SET LOGGER
    if params['experiment']['logging']:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=params['experiment']['experiment_folder'],
                                                 name=params['experiment']['sub_folder'],
                                                 version=version, log_graph=True)
    else:
        tb_logger = False

    if params['train']['early_stopping']:
        early_stop_callback = EarlyStopping(monitor="val_loss",
                                            patience=params['train']['patience'],
                                            mode="min")
        callback_funcs = [checkpoint_callback, early_stop_callback]
    else:
        callback_funcs = [checkpoint_callback]

    trainer = pl.Trainer(devices=gpus, max_epochs=max_epochs,
                         gradient_clip_val=params['train']['gradient_clip_val'],
                         gradient_clip_algorithm=params['train']['gradient_clip_algorithm'],
                         accelerator="gpu",
                         callbacks=callback_funcs, logger=tb_logger,
                         #profiler='simple',
                         precision=params['experiment']['precision'],
                         #strategy=DDPStrategy(find_unused_parameters=False),
                         strategy="ddp",
                         #limit_train_batches=0.1, limit_val_batches=0.1,  # for debug
                         #fast_dev_run=True, # for debug, run the model code once quickly
                        )

    return trainer

def do_predict(trainer, model, predict_params, test_data):
    scores = trainer.predict(model, dataloaders=test_data)
    scores = torch.concat(scores)
    tensor_to_submission_file(scores, predict_params)

def do_test(trainer, model, test_data):
    outputs = trainer.test(model, dataloaders=test_data)

def train(params, gpus, mode, checkpoint_path, model_name, options):
    """ main training/evaluation method
    """
    # ------------
    # model & data
    # ------------
    get_cuda_memory_usage(gpus)

    if mode in ['val', 'predict', 'heldout']:
        params['train']['thresholds']['combine'] = False

        if options.thresholds is not None:
            thresholds = options.thresholds
            params['train']['thresholds']['boxi_0015'] = thresholds[0]
            params['train']['thresholds']['boxi_0034'] = thresholds[1]
            params['train']['thresholds']['boxi_0076'] = thresholds[2]

        if options.nomask:
            print(f'no binary mask the prediction {options.nomask}')
            params['train']['nomask'] = options.nomask

    print(f"mode: {mode}\nthresholds: {params['train']['thresholds']}")

    data = DataModule(params['dataset'], params['train'], mode)
    model = load_model(model_name, params, checkpoint_path)
    # ------------
    # Add your models here
    # ------------

    # ------------
    # trainer
    # ------------
    trainer = get_trainer(gpus, params, options)
    get_cuda_memory_usage(gpus)
    # ------------
    # train & final validation
    # ------------
    if mode == 'train':
        print("------------------")
        print("--- TRAIN MODE ---")
        print("------------------")
        trainer.fit(model, data)


    if mode == "val":
    # ------------
    # VALIDATE
    # ------------
        print("---------------------")
        print("--- VALIDATE MODE ---")
        print("---------------------")
        do_test(trainer, model, data.val_dataloader())


    if mode == 'predict' or mode == 'heldout':
    # ------------
    # PREDICT
    # ------------
        print("--------------------")
        print("--- PREDICT MODE ---")
        print("--------------------")
        print("REGIONS!:: ", params["dataset"]["regions"], params["predict"]["region_to_predict"])
        if params["predict"]["region_to_predict"] not in params["dataset"]["regions"]:
            print("EXITING... \"regions\" and \"regions to predict\" must indicate the same region name in your config file.")
        else:
            do_predict(trainer, model, params["predict"], data.test_dataloader())

    get_cuda_memory_usage(gpus)

def update_params_based_on_args(options):
    config_p = os.path.join('configs', options.config_path)
    params = load_config(config_p)

    if options.loss is not None:
        params['train']['loss'] = options.loss

    if options.name != '':
        print(params['experiment']['name'])
        params['experiment']['name'] = options.name

    if options.sub_folder is not None:
        params['experiment']['sub_folder'] = options.sub_folder

    if options.experiment_folder is not None:
        params['experiment']['experiment_folder'] = options.experiment_folder

    if options.sigmoid is not None:
        params['train']['sigmoid'] = options.sigmoid

    return params

def set_parser():
    """ set custom parser """

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str, help='model name')
    parser.add_argument("-project", "--project", type=str, required=False, default='weather4cast-2022',
                        help="the name of project")
    parser.add_argument("-f", "--config_path", type=str, required=False, default='./configs/config_basline.yaml',
                        help="path to config-yaml")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=1,
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train',
                        help="choose mode: train (default) / val / predict")
    parser.add_argument("-l", "--loss", type=str, required=False, default=None,
                        help="the loss function")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='',
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=False, default='',
                         help="Set the name of the experiment")
    parser.add_argument("-p", "--pack_name", type=str, required=False, default=None,
                         help="the pack name of prediction")
    parser.add_argument("-savedir", "--savedir", type=str, required=False, default=None,
                         help="the path of prediciton")
    parser.add_argument("-r", "--regions", type=str, required=False, default='boxi_0015,boxi_0034,boxi_0076,roxi_0004,roxi_0005,roxi_0006,roxi_0007',
                         help="the regions of prediction")
    parser.add_argument("-y", "--years", type=str, required=False, default='2019,2020',
                         help="the years of prediction")
    parser.add_argument("-t", "--thresholds", type=float, nargs='+', required=False, default=None,
                         help="the rain thresholds corresponding to regions, such as boxi_0015, boxi_0034, boxi_0076")
    parser.add_argument("-s", "--sub_folder", type=str, required=False, default=None,
                         help="logging sub-folder")
    parser.add_argument("-shuffle_sample", "--shuffle_sample", type=ast.literal_eval, required=False, default=None,
                        help="whether shuffle sample or not")
    parser.add_argument("-e", "--experiment_folder", type=str, required=False, default=None,
                         help="folder to save logs")
    parser.add_argument("-nomask", "--nomask", type=ast.literal_eval, required=False, default=None,
                        help="whether mask the prediction or not")
    parser.add_argument("-sigmoid", "--sigmoid", type=ast.literal_eval, required=False, default=None,
                        help="whether add sigmoid activation function")

    return parser


def pack_prediction(options):
    print(f'pack the prediction of the {options.regions} regions to {options.pack_name}.zip...')
    os.system(f'cd {options.savedir} && zip -r {options.pack_name}.zip 20*')


def main():
    parser = set_parser()
    options = parser.parse_args()

    params = update_params_based_on_args(options)

    if options.mode == 'predict' or options.mode == 'heldout':
        if options.savedir is not None:
            params['predict']['submission_out_dir'] = options.savedir

        for year in options.years.split(','):
            for reg in options.regions.split(','):
                params['dataset']['regions'] = [reg]
                params['dataset']['years'] = [year]
                params['predict']['year_to_predict'] = int(year)
                params['predict']['region_to_predict'] = reg
                train(params, options.gpus, options.mode, options.checkpoint, options.model_name, options)
                os.system(f"gzip -9f {params['predict']['submission_out_dir']}/{year}/{reg}.pred.h5")

        if options.pack_name is not None:
            pack_prediction(options)
        else:
            print('skipping the pack step, please manually pack the prediction...')
    else:
        if options.mode == 'val':
            if params['dataset']['regions'] != options.regions.split(','):
                params['dataset']['regions'] = options.regions.split(',')
            print(params['dataset']['regions'])
        train(params, options.gpus, options.mode, options.checkpoint, options.model_name, options)


if __name__ == "__main__":
    main()
    """ examples of usage:

    1) train from scratch on one GPU
    python train.py unet3d --gpus 2 --mode train --config_path config_baseline.yaml --name baseline_train

    2) train from scratch on four GPUs
    python train.py unet3d --gpus 0 1 2 3 --mode train --config_path config_baseline.yaml --name baseline_train

    3) fine tune a model from a checkpoint on one GPU
    python train.py unet3d --gpus 1 --mode train  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_tune

    4) evaluate a trained model from a checkpoint on two GPUs
    python train.py unet3d --gpus 0 1 --mode val  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_validate

    5) generate predictions (plese note that this mode works only for one GPU)
    python train.py unet3d --gpus 1 --mode predict  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"

    6) generate predictions for the held-out dataset (plese note that this mode works only for one GPU)
    python train.py --gpus 1 --mode heldout  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"

    """
