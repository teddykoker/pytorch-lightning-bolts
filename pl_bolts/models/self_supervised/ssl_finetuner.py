import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plm
from pl_bolts.models.self_supervised import SSLEvaluator
import torch
import torch.nn.functional as F


class SSLFineTuner(pl.LightningModule):

    def __init__(self, backbone, in_features, num_classes, hidden_dim=1024):
        """
        Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP
        with 1024 units

        Example::

            from pl_bolts.utils.self_supervised import SSLFineTuner
            from pl_bolts.models.self_supervised import CPCV2
            from pl_bolts.datamodules import CIFAR10DataModule
            from pl_bolts.models.self_supervised.cpc.transforms import CPCEvalTransformsCIFAR10,
                                                                        CPCTrainTransformsCIFAR10

            # pretrained model
            backbone = CPCV2.load_from_checkpoint(PATH, strict=False)

            # dataset + transforms
            dm = CIFAR10DataModule(data_dir='.')
            dm.train_transforms = CPCTrainTransformsCIFAR10()
            dm.val_transforms = CPCEvalTransformsCIFAR10()

            # finetuner
            finetuner = SSLFineTuner(backbone, in_features=backbone.z_dim, num_classes=backbone.num_classes)

            # train
            trainer = pl.Trainer()
            trainer.fit(finetuner, dm)

            # test
            trainer.test(datamodule=dm)

        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()
        self.backbone = backbone
        self.ft_network = SSLEvaluator(
            n_input=in_features,
            n_classes=num_classes,
            p=0.2,
            n_hidden=hidden_dim
        )

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        result = pl.TrainResult(loss)
        result.log('train_acc', acc, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss}, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        result = pl.EvalResult()
        result.log_dict({'test_acc': acc, 'test_loss': loss})
        return result

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.backbone(x)
        feats = feats.view(x.size(0), -1)
        logits = self.ft_network(feats)
        loss = F.cross_entropy(logits, y)
        acc = plm.accuracy(logits, y)

        return loss, acc

    def configure_optimizers(
        self,
    ):
        return torch.optim.Adam(self.ft_network.parameters(), lr=0.0002)
