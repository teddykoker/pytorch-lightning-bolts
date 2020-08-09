import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchlars import LARS

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.metrics import mean, accuracy

from pl_bolts.models.self_supervised.evaluator import SSLEvaluator, Flatten
from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, stl10_normalization, imagenet_normalization


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class LinearEval(nn.Module):
    def __init__(self, input_dim=2048, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.num_classes, bias=True))

    def forward(self, x):
        return self.model(x)


class SimCLR(pl.LightningModule):
    def __init__(self,
                 datamodule: pl.LightningDataModule = None,
                 data_dir: str = './',
                 learning_rate: float = 0.00006,
                 weight_decay: float = 0.0005,
                 input_height: int = 32,
                 batch_size: int = 128,
                 online_ft: bool = False,
                 num_workers: int = 4,
                 optimizer: str = 'lars',
                 lr_sched_step: float = 30.0,
                 lr_sched_gamma: float = 0.5,
                 lars_momentum: float = 0.9,
                 lars_eta: float = 0.001,
                 loss_temperature: float = 0.5,
                 **kwargs):
        """
        PyTorch Lightning implementation of `SIMCLR <https://arxiv.org/abs/2002.05709.>`_

        Paper authors: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton.

        Model implemented by:

            - `William Falcon <https://github.com/williamFalcon>`_
            - `Tullie Murrell <https://github.com/tullie>`_

        Example:

            >>> from pl_bolts.models.self_supervised import SimCLR
            ...
            >>> model = SimCLR()

        Train::

            trainer = Trainer()
            trainer.fit(model)

        CLI command::

            # cifar10
            python simclr_module.py --gpus 1

            # imagenet
            python simclr_module.py
                --gpus 8
                --dataset imagenet2012
                --data_dir /path/to/imagenet/
                --meta_dir /path/to/folder/with/meta.bin/
                --batch_size 32

        Args:
            datamodule: The datamodule
            data_dir: directory to store data
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            online_ft: whether to tune online or not
            num_workers: number of workers
            optimizer: optimizer name
            lr_sched_step: step for learning rate scheduler
            lr_sched_gamma: gamma for learning rate scheduler
            lars_momentum: the mom param for lars optimizer
            lars_eta: for lars optimizer
            loss_temperature: float = 0.
        """
        super().__init__()
        self.save_hyperparameters()
        self.online_ft = online_ft

        self.loss_func = self.init_loss()
        self.encoder = self.init_encoder()
        self.projection = self.init_projection()

        # init default datamodule
        if datamodule is None:
            datamodule = CIFAR10DataModule(data_dir, num_workers=num_workers, batch_size=batch_size)
            normalize = cifar10_normalization() if self.hparams.normalize_data else None

            datamodule.train_transforms = SimCLRTrainDataTransform(
                input_height,
                jitter_strength=0.5,
                gaussian_blur=False,
                normalize=normalize
            )
            datamodule.val_transforms = SimCLREvalDataTransform(
                input_height,
                normalize=normalize
            )

            # modify for cifar10
            self.encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.datamodule = datamodule

        # TODO: set scheduler params for each step instead of epoch

        if self.online_ft:
            self.online_evaluator = LinearEval(num_classes=self.datamodule.num_classes)

    def init_loss(self):
        return nt_xent_loss

    def init_encoder(self):
        return resnet50()

    def init_projection(self):
        return Projection()

    def forward(self, x):
        return self.encoder(x)[-1]

    def training_step(self, batch, batch_idx):
        if isinstance(self.datamodule, STL10DataModule):
            labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), y = batch
        h1 = self.forward(img_1)
        h2 = self.forward(img_2)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        # return h1, z1, h2, z2
        loss = self.loss_func(z1, z2, self.hparams.loss_temperature)
        log = {'train_ntx_loss': loss}

        # don't use the training signal, just finetune the MLP to see how we're doing downstream
        if self.online_ft:
            if isinstance(self.datamodule, STL10DataModule):
                (img_1, img_2), y = labeled_batch

            with torch.no_grad():
                h1 = self.forward(img_1)
                z1 = self.projection(h1)

            # no grads to unsupervised encoder
            h_feats = h1.detach()

            preds = self.online_evaluator(h_feats)
            mlp_loss = F.cross_entropy(preds, y)

            loss = loss + mlp_loss
            log['train_mlp_loss'] = mlp_loss

        result = pl.TrainResult(minimize=loss)
        result.log_dict(log)

        return result

    def validation_step(self, batch, batch_idx):
        if isinstance(self.datamodule, STL10DataModule):
            labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), y = batch
        h1 = self.forward(img_1)
        h2 = self.forward(img_2)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.loss_func(z1, z2, self.hparams.loss_temperature)
        result = {'val_loss': loss}

        if self.online_ft:
            if isinstance(self.datamodule, STL10DataModule):
                (img_1, img_2), y = labeled_batch
                h1, z1 = self.forward(img_1)

            preds = self.online_evaluator(h1)
            mlp_loss = F.cross_entropy(preds, y)

            acc = accuracy(preds, y)
            result['mlp_acc'] = acc
            result['mlp_loss'] = mlp_loss

        return result

    def validation_epoch_end(self, outputs: list):
        val_loss = mean(outputs, 'val_loss')

        result = pl.EvalResult(checkpoint_on=val_loss)

        log = dict(val_loss=val_loss)
        if self.online_ft:
            mlp_acc = mean(outputs, 'mlp_acc')
            mlp_loss = mean(outputs, 'mlp_loss')

            log['val_mlp_acc'] = mlp_acc
            log['val_mlp_loss'] = mlp_loss

        result.log_dict(log)

        return result

    # TODO: separate opt, scheduler for online eval
    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'lars':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.lars_momentum,
                weight_decay=self.hparams.weight_decay,
            )

            optimizer = LARS(
                optimizer=optimizer,
                eps=1e-8,
                trust_coefficient=self.hparams.trust_coef,
                clip=False
            )
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer}')

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.warmup_start_lr,
            eta_min=self.hparams.eta_min
        )

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return optimizer, scheduler

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--online_ft', action='store_true', help='run online finetuner')
        parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, imagenet2012, stl10')

        (args, _) = parser.parse_known_args()
        # Data
        parser.add_argument('--data_dir', type=str, default='.')
        parser.add_argument('--normalize_data', type=bool, default=False)

        # Training
        parser.add_argument('--gpus', type=int, default=2)
        parser.add_argument('--sync_batchnorm', type=bool, default=True)
        parser.add_argument('--distributed_backend', type=str, default='ddp')

        parser.add_argument('--optimizer', choices=['adam', 'lars'], default='lars')
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--num_workers', default=16, type=int)
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--lars_momentum', type=float, default=0.9)
        parser.add_argument('--trust_coef', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=1e-6)

        parser.add_argument('--warmup_epochs', type=int, default=10)
        parser.add_argument('--max_epochs', type=int, default=100)
        parser.add_argument('--warmup_start_lr', type=float, default=0.)
        parser.add_argument('--eta_min', type=float, default=0.)

        # Model
        parser.add_argument('--loss_temperature', type=float, default=0.5)
        parser.add_argument('--meta_dir', default='.', type=str, help='path to meta.bin for imagenet')

        return parser


# todo: covert to CLI func and add test
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # model checkpointing callback
    checkpoint_callback = ModelCheckpoint(verbose=True, save_last=True, save_top_k=3)
    lr_logger = LearningRateLogger()
    logger = WandbLogger(project='simclr')

    # model checkpointing callback
    checkpoint_callback = ModelCheckpoint(verbose=True, save_last=True, save_top_k=3)
    lr_logger = LearningRateLogger()
    logger = WandbLogger(project='simclr')

    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    datamodule = None
    if args.dataset == 'stl10':
        datamodule = STL10DataModule.from_argparse_args(args)
        datamodule.train_dataloader = datamodule.train_dataloader_mixed
        datamodule.val_dataloader = datamodule.val_dataloader_mixed

        (c, h, w) = datamodule.size()
        datamodule.train_transforms = SimCLRTrainDataTransform(h)
        datamodule.val_transforms = SimCLREvalDataTransform(h)

    elif args.dataset == 'imagenet2012':
        datamodule = ImagenetDataModule.from_argparse_args(args, image_size=196)
        (c, h, w) = datamodule.size()
        datamodule.train_transforms = SimCLRTrainDataTransform(h)
        datamodule.val_transforms = SimCLREvalDataTransform(h)

    model = SimCLR(**args.__dict__, datamodule=datamodule)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[lr_logger]
    )
    trainer.fit(model)

"""
TODOs:

scheduler correct steps

1. exclude bn and bias terms
3. opt for online
4. offline eval
5. LR formula for lars
"""