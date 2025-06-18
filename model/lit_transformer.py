import lightning as L
import numpy as np
import torch
from torch.optim import Adam

from model.transformer import transformer
from utils.loss import NormalizedL2Loss
from utils.y_params import YParams


class VisionTransformerLightning(L.LightningModule):
    """
    Vision Transformer Lightning Module for training with PyTorch Lightning.

    This class is a custom PyTorch Lightning module for the Vision Transformer,
    including methods for forward pass, training step, validation step, and
    configuring optimizers.

    Attributes:
        model: Transformer model loaded with the given parameters.
        loss_func: Loss function used for training.
        params: Training and model configuration parameters.
        lr: Learning rate for the optimizer.
        batch_size: Batch size used for training.
    """

    def __init__(self, params: YParams) -> None:
        super().__init__()
        self.model = transformer(params)
        self.params = params

        self.lr = params.lr
        self.batch_size = params.global_batch_size
        self.normalized_l2_loss = NormalizedL2Loss()
        self.training_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the model.

        :param x: Input tensor to the transformer model.
        :return: Output tensor from the model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        :param batch: The input and target batch data.
        :param batch_idx: Index of the current batch.
        :return: Computed loss for the current training batch.
        """
        inp, tar = batch
        gen = self(inp)

        loss = self.normalized_l2_loss(gen, tar)

        self.training_step_outputs.append(loss)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        :param batch: The input and target batch data.
        :param batch_idx: Index of the current batch.
        """
        inp, tar = batch
        gen = self(inp)
        val_loss = self.normalized_l2_loss(gen, tar)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        """
        Test step for evaluating the model on the test set.

        :param batch: The input and target batch data from the test DataLoader.
        :param batch_idx: Index of the current batch in the test set.
        :return: Computed metrics or outputs for the current test batch.
        """
        inp, tar = batch
        gen = self(inp)

        loss = self.normalized_l2_loss(gen, tar)

        # Log the test loss; logging for test metrics is often done on epoch level
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))

        if self.params.lr_schedule == 'cosine':
            if self.params.warmup > 0:
                scheduler = {
                        'scheduler': torch.optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lambda x: min(
                                (x + 1) / self.params.warmup,
                                0.5 * (1 + np.cos(np.pi * x / self.params.num_iters))
                            )
                        ),
                        'name':      'warmup_cosine',
                        'interval':  self.params.learning_rate_logging_interval,
                        'frequency': self.params.learning_rate_logging_frequency
                }
            else:
                scheduler = {
                        'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params.num_iters),
                        'name':      'cosine_annealing',
                        'interval':  'epoch',
                        'frequency': 1
                }

            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return {"optimizer": optimizer}
