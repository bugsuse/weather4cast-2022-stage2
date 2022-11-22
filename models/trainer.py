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


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix

from utils.metrics import *
from utils.evaluate import *
from utils.losses import get_lossfx
from utils.upsample import upsample
from utils.weight import group_weight
from . import get_model


class ModelTrainer(pl.LightningModule):
    def __init__(self, model_name: str, model_params: dict, params: dict,
                 **kwargs):
        super(ModelTrainer, self).__init__()

        self.model_name = model_name
        self.model = get_model(model_name, model_params)

        self.save_hyperparameters()
        self.params = params

        self.main_metric = 'BCE with logits' #mse [log(y+1)-yhay]'
        pos_weight = torch.tensor(params['pos_weight'])

        self.loss = params['loss']
        self.bs = params['batch_size']
        self.loss_fn = get_lossfx(self.loss, params)

        self.freeze_model_params()

        t = f"============== n_workers: {params['n_workers']} | batch_size: {params['batch_size']} \n"+\
            f"============== loss: {self.loss} | weight: {pos_weight} (if using BCEwLL)"
        print(t)

    def on_fit_start(self):
        """ create a placeholder to save the results of the metric per variable """
        metric_placeholder = {self.main_metric: -1}
        self.logger.log_hyperparams(self.hparams, metric_placeholder)

    def forward(self, x):
        if self.params.get('reshape') is not None and self.params.get('reshape'):
            bs, ch, ts, w, h = x.shape
            x = x.reshape(bs, -1, w, h)

        if self.params.get('interp_inp') is not None and self.params.get('interp_inp'):
             if self.params.get('reshape') is not None and self.params.get('reshape'):
                 x = upsample(x, self.params).squeeze()
             else:
                 bs, ch, ts, w, h = x.shape
                 height = int(h * self.params['scale_factor'])
                 width = int(w * self.params['scale_factor'])
                 tmp = torch.zeros((bs, ch, ts, width, height), dtype=self.dtype, device=self.device)
                 for i in range(ts):
                     tmp[:, :, i] = upsample(x[:, :, i], self.params)[:, 0]
                 x = tmp.half()

        if self.model_name == 'earthformer':
            x = x.permute(0, 2, 3, 4, 1)

        x = self.model(x)

        if self.model_name == 'earthformer':
            x = x.permute(0, 4, 1, 2, 3)

        if self.params.get('sigmoid') is not None and self.params.get('sigmoid'):
            x = nn.Sigmoid()(x)

        if self.params.get('padding') is not None:
            pad = self.params.get('padding')
            x = x[..., pad:-pad, pad:-pad]

        return x

    def retrieve_only_valid_pixels(self, x, m):
        """ we asume 1s in mask are invalid pixels """
        return x[~m]

    def get_target_mask(self, metadata):
        mask = metadata['target']['mask']
        return mask

    def _compute_loss(self, y_hat, y, agg=True, mask=None):
        if mask is not None:
            y_hat = self.retrieve_only_valid_pixels(y_hat, mask)
            y = self.retrieve_only_valid_pixels(y, mask)

        if agg:
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')

        return loss

    def training_step(self, batch, batch_idx, phase='train'):
        x, y, metadata  = batch

        y_hat = self.forward(x)
        y_hat = upsample(y_hat, self.params) if self.params.get('interp_inp') is None or not self.params.get('interp_inp') else y_hat

        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)

        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, phase='val'):
        x, y, metadata = batch
        regions = metadata['region']

        y_hat = self.forward(x)
        y_hat = upsample(y_hat, self.params) if self.params.get('interp_inp') is None or not self.params.get('interp_inp') else y_hat

        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)

        y_hat = mask_evaluation(y_hat, y, regions, self.params, 'valid')

        values = combine_metrics(y, y_hat, phase)
        #LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)
        self.log_dict(values, on_step=False, on_epoch=True, batch_size=self.bs, sync_dist=True)

        return loss

    def validation_epoch_end(self, outputs, phase='val'):
        avg_loss = torch.stack([x for x in outputs]).mean()

        self.log(f'{phase}_loss_epoch', avg_loss, prog_bar=True, batch_size=self.bs, sync_dist=True)
        self.log(self.main_metric, avg_loss, batch_size=self.bs, sync_dist=True)

    def test_step(self, batch, batch_idx, phase='test'):
        x, y, metadata = batch
        regions = metadata['region']

        y_hat = self.forward(x)
        y_hat = upsample(y_hat, self.params) if self.params.get('interp_inp') is None or not self.params.get('interp_inp') else y_hat

        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)

        y_hat = mask_evaluation(y_hat, y, regions, self.params, 'valid')

        values = combine_metrics(y, y_hat, phase)
        #LOGGING
        self.log(f'{phase}_loss', loss, prog_bar=True, batch_size=self.bs, sync_dist=True)
        self.log_dict(values, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.bs, sync_dist=True)

        outputs = {'target': y, 'pred': y_hat}

        return outputs

    def test_epoch_end(self, outputs, phase='val'):
        pass

    def predict_step(self, batch, batch_idx, phase='predict'):
        x, y, metadata = batch
        regions = metadata['region']
        y_hat = self.forward(x)
        y_hat = upsample(y_hat, self.params) if self.params.get('interp_inp') is None or not self.params.get('interp_inp') else y_hat

        mask = self.get_target_mask(metadata)

        if self.params.get('nomask') is not None:
            return y_hat
        else:
            y_hat = mask_evaluation(y_hat, y, regions, self.params, 'predict')

            return y_hat

    def configure_optimizers(self):
        optim_params = self.params[self.params['optim']]

        model_parameters = self.set_model_params_optimizer()

        if self.params['optim'].lower() == 'adam':
            optimizer = optim.Adam(model_parameters, lr=float(self.params["lr"]), **optim_params)
        elif self.params['optim'].lower() == 'adamw':
            optimizer = optim.AdamW(model_parameters, lr=float(self.params["lr"]), **optim_params)
        elif self.params['optim'].lower() == 'sgd':
            optimizer = optim.SGD(model_parameters, lr=float(self.params["lr"]), **optim_params)
        else:
            raise ValueError(f'No support {self.params.optim} optimizer!')

        ## configure scheduler
        lr_params = self.params[self.params['scheduler']]

        print("Learning rate:", self.params["lr"],
              "optimizer: ", self.params["optim"], "optimier parameters: ", optim_params,
              "scheduler: ", self.params['scheduler'], "scheduler paramsters: ", lr_params)

        if self.params['scheduler'] == 'exp':
            scheduler = lr_scheduler.ExponentialLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'cosinewarm':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'onecycle':
            scheduler = lr_scheduler.OneCycleLR(optimizer, **lr_params)
            return [optimizer], [scheduler]
        elif self.params['scheduler'] == 'reducelr':
            return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer, **lr_params),
                        'monitor': self.params['reducelr_monitor'],
                    }
                }
        else:
            raise ValueError(f"No support {self.params['scheduler']} scheduler!")


    def set_model_params_optimizer(self):
        if 'no_bias_decay' in self.params and self.params.get('no_bias_decay'):
            if 'encoder_lr_ratio' in self.params:
                encoder_lr_ratio = self.params.get('encoder_lr_ratio')
                group_decay_encoder, group_no_decay_encoder = group_weight(self.model.down_convs)
                group_decay_decoder, group_no_decay_decoder = group_weight(self.model.up_convs)
                base_lr = self.params['lr']
                params = [{'params': group_decay_decoder},
                          {'params': group_no_decay_decoder, 'weight_decay': 0.0},
                          {'params': group_decay_encoder, 'lr': base_lr * encoder_lr_ratio},
                          {'params': group_no_decay_encoder, 'lr': base_lr * encoder_lr_ratio, 'weight_decay': 0.0}]
                print(f'separately set lr with no_bias_decay for encoder {base_lr} and decoder {base_lr * encoder_lr_ratio}...')
            else:
                group_decay, group_no_decay = group_weight(self.model)
                params = [{'params': group_decay},
                          {'params': group_no_decay, 'weight_decay': 0.0}]
                print(f'set params with no_bias_decay...')
        elif 'encoder_lr_ratio' in self.params:
            encoder_lr_ratio = self.params.get('encoder_lr_ratio')
            base_lr = float(self.params['lr'])
            print(encoder_lr_ratio, base_lr)
            print(f'separately set lr for encoder {base_lr} and decoder {base_lr * encoder_lr_ratio}...')
            params = [{'params': self.model.up_convs.parameters()},
                      {'params': self.model.reduce_channels.parameters()},
                      {'params': self.model.down_convs.parameters(), 'lr': base_lr * encoder_lr_ratio}]
        else:
            params = self.model.parameters()

        return params


    def freeze_model_params(self):
        if 'freeze_encoder' in self.params and self.params.get('freeze_encoder'):
            print('freezing the parameters of encoder...')
            for name, param in self.model.down_convs.named_parameters():
                param.requires_grad = False

        if 'freeze_decoder' in self.params and self.params.get('freeze_decoder'):
            print('freezing the parameters of decoder...')
            for name, param in self.model.down_convs.named_parameters():
                param.requires_grad = False

        if 'freeze_output' in self.params and self.params.get('freeze_output'):
            print('freezing the parameters of final output...')
            for name, param in self.model.reduce_channels.named_parameters():
                param.requires_grad = False


    def seq_metrics(self, y_true, y_pred):
        text = ''
        cm = confusion_matrix(y_true, y_pred).ravel()
        if len(cm)==4:
            tn, fp, fn, tp = cm
            recall, precision, F1 = 0, 0, 0

            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            r = f'r: {recall:.2f}'

            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            p = f'p: {precision:.2f}'

            if (precision + recall) > 0:
                F1 = 2 * (precision * recall) / (precision + recall)
            f = f'F1: {F1:.2f}'

            acc = (tn + tp) / (tn+fp+fn+tp)
            text = f"{r} | {p} | {f} | acc: {acc} "

        return text

def main():
    print("running")

if __name__ == 'main':
    main()

