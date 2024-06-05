from copy import deepcopy

import numpy
import pandas
from lightning.pytorch import Trainer
from pytorch_forecasting import (
    MultiLoss,
    NBeats,
    QuantileLoss,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)

random_generator = numpy.random.default_rng(seed=0)

sample_data = pandas.DataFrame(
    {
        "endogenous_variable_1": random_generator.uniform(low=-10, high=10, size=10000),
        # "endogenous_variable_2": random_generator.uniform(low=-10, high=10, size=10000),
        "series_identifier": numpy.repeat(numpy.arange(400), 25),
        "temporal_identifier": numpy.tile(numpy.arange(25), 400),
    }
)

sample_data

training_dataset = TimeSeriesDataSet(
    sample_data,
    "temporal_identifier",
    "endogenous_variable_1",
    ["series_identifier"],
    max_encoder_length=20,
    min_encoder_length=20,
    min_prediction_idx=20,
    max_prediction_length=5,
    min_prediction_length=5,
    time_varying_unknown_reals=["endogenous_variable_1"],
)

training_dataset.get_parameters()

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, sample_data, stop_randomization=True, predict=True
)

validation_dataset.get_parameters()

training_data_loader = training_dataset.to_dataloader(
    train=True, batch_size=5, num_workers=0
)

validation_data_loader = validation_dataset.to_dataloader(
    train=False, batch_size=5, num_workers=0
)


forecaster = NBeats.from_dataset(
    training_dataset,
    log_val_interval=10,
    # lstm_layers=2,
    # output_size=5,
    # loss=MultiLoss([QuantileLoss(quantiles=[0.025, 0.05, 0.5, 0.95, 0.975])]),
)

forecaster.hparams
forecaster_copy = deepcopy(forecaster)

pytorch_trainer = Trainer(accelerator="cpu", max_epochs=500, min_epochs=2)

pytorch_trainer.fit(
    forecaster,
    train_dataloaders=training_data_loader,
    val_dataloaders=validation_data_loader,
)

for p, p_c in zip(forecaster.parameters(), forecaster_copy.parameters()):
    print((p - p_c).abs().sum())
