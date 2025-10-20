import os
import logging
import shutil

import mlflow 
from mlflow.exceptions import MlflowException
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch.loggers as pl_loggers

from raag_midi_gen.paths import EXPERIMENT_LOGS_DIR, s3_bucket_name


class ModelCheckpointWithCleanupCallback(ModelCheckpoint):
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        shutil.rmtree(os.path.dirname(os.path.dirname(self.dirpath)))


def get_lightning_mlflow_logger(experiment_name: str, entrypoint_script: str, git_hash: str) -> pl_loggers.MLFlowLogger:    
    return pl_loggers.MLFlowLogger(
        experiment_name=experiment_name,
        tags={
            'entrypoint_script': entrypoint_script,
            'git_hash': git_hash
        },
        tracking_uri=os.path.join(EXPERIMENT_LOGS_DIR, './mlruns'),
        log_model=True,
        artifact_location=f's3://{s3_bucket_name}/{experiment_name}/'
    )


def setup_sklearn_mlflow(experiment_name: str) -> str:    
    mlflow.set_tracking_uri(os.path.join(EXPERIMENT_LOGS_DIR, './mlruns'))
    
    try:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=f's3://{s3_bucket_name}/{experiment_name}/'
        )
    except MlflowException:
        logging.info(f'Experiment {experiment_name} already exists.')
    finally:
        mlflow.set_experiment(experiment_name)

    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True
    )

    return mlflow