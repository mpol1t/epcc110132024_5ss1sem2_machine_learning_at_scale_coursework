import logging
import time
from typing import List

from lightning import Trainer, Callback
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, DeviceStatsMonitor, ModelSummary, \
    BatchSizeFinder, LearningRateFinder
from lightning.pytorch.loggers import CSVLogger

from model.lit_transformer import VisionTransformerLightning
from utils import logging_utils
from utils.data_loader import ERA5DataModule
from utils.parser import get_params
from utils.y_params import YParams

logging_utils.config_logger()


def train_model(params: YParams) -> None:
    """
    Train the Vision Transformer model using the provided parameters.

    :param params: An instance of YParams containing training parameters.
    """
    model: VisionTransformerLightning = VisionTransformerLightning(params)
    data_module: ERA5DataModule = ERA5DataModule(params)

    csv_logger = CSVLogger(save_dir=params.exp_dir, name=params.exp_name)

    callbacks: List[Callback] = [
            DeviceStatsMonitor(),
            EarlyStopping(
                mode=params.early_stopping_mode,
                strict=params.early_stopping_strict,
                monitor=params.early_stopping_monitor,
                verbose=params.early_stopping_verbose,
                patience=params.early_stopping_patience,
                min_delta=params.early_stopping_min_delta,
                check_finite=params.early_stopping_check_finite,
                log_rank_zero_only=params.early_stopping_log_rank_zero_only
            ),
            ModelSummary(max_depth=params.trainer_model_summary_max_depth),
            LearningRateMonitor(logging_interval=params.learning_rate_logging_interval)
    ]

    if params.tune_batch_size_enabled:
        callbacks.append(
            BatchSizeFinder(
                mode=params.tune_batch_size_mode,
                init_val=params.tune_batch_size_init_val,
                max_trials=params.tune_batch_size_max_trials,
                batch_arg_name=params.tune_batch_size_arg_name,
                steps_per_trial=params.tune_batch_size_steps_per_trial
            )
        )

    if params.tune_learning_rate_enabled:
        callbacks.append(
            LearningRateFinder(
                mode=params.tune_learning_rate_mode,
                min_lr=params.tune_learning_rate_min_lr,
                max_lr=params.tune_learning_rate_max_lr,
                attr_name=params.tune_learning_rate_attr_name,
                update_attr=params.tune_learning_rate_update_attr,
                num_training_steps=params.tune_learning_rate_num_training_steps,
                early_stop_threshold=params.tune_learning_rate_early_stop_threshold
            )
        )

    trainer: Trainer = Trainer(
        max_epochs=params.epochs,
        log_every_n_steps=params.log_every_n_steps,
        devices=params.devices,
        num_nodes=params.num_nodes,
        accelerator=params.accelerator,
        precision=params.precision if params.quantize else 32,
        strategy=params.strategy,
        default_root_dir=params.exp_dir,
        logger=csv_logger,
        use_distributed_sampler=params.trainer_use_distributed_sampler,
        sync_batchnorm=params.trainer_sync_batch_norm,
        gradient_clip_algorithm=params.trainer_gradient_clip_algorithm,
        accumulate_grad_batches=params.trainer_accumulate_grad_batches,
        benchmark=params.trainer_benchmark,
        enable_model_summary=params.trainer_enable_model_summary,
        callbacks=callbacks
    )

    start_time: float = time.time()

    trainer.fit(model, datamodule=data_module)

    logging.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Trainer for testing on a single device
    test_trainer: Trainer = Trainer(
        devices=1,  # Use a single device
        num_nodes=1,  # Single node (non-distributed)
        accelerator=params.accelerator,
        precision=params.precision if params.quantize else 32,
        logger=False,  # Optionally disable logging for testing
        callbacks=[],  # Optionally, no callbacks during testing
    )

    # Later, when you want to test your model:
    test_trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    train_model(get_params())
