import time
from copy import deepcopy

import numpy
import pandas
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import (
    DeepAR,
    MultiLoss,
    NBeats,
    NHiTS,
    QuantileLoss,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)

# which model to use
# model_class = TemporalFusionTransformer
# model_class = NBeats
model_class = NHiTS
model_class = DeepAR

# False = no log - pass, 500 epoch for less than 5s (TFT)
# True = do log - fail (TFT)
log_val = True

# False = no X is passed - fail if log_val (TFT)
# True = X is passed - pass, 5m for just 1 epoch (TFT)
X_data = False


random_generator = numpy.random.default_rng(seed=0)

multi_index = pandas.MultiIndex.from_frame(
    pandas.DataFrame(
        {
            "series_identifier": numpy.repeat(numpy.arange(4000), 25),
            "temporal_identifier": numpy.tile(numpy.arange(25), 4000),
        }
    )
)
X = (
    pandas.DataFrame(
        {
            "exogenous_variable_1": random_generator.uniform(
                low=-10, high=10, size=100000
            ),
        }
    ).set_index(multi_index)
    if X_data
    else None
)
y = pandas.DataFrame(
    {
        "endogenous_variable_1": random_generator.uniform(
            low=-10, high=10, size=100000
        ),
    }
).set_index(multi_index)


if X is not None:
    sample_data = X.join(y, on=["series_identifier", "temporal_identifier"])
else:
    sample_data = deepcopy(y)
sample_data.reset_index(level=[0, 1], inplace=True)
print(sample_data.columns)

training_dataset = TimeSeriesDataSet(
    data=sample_data,
    time_idx="temporal_identifier",
    target="endogenous_variable_1",
    group_ids=["series_identifier"],
    max_encoder_length=20,
    min_encoder_length=20,
    min_prediction_idx=20,
    max_prediction_length=5,
    min_prediction_length=5,
    time_varying_unknown_reals=["endogenous_variable_1"],
    time_varying_known_reals=[] if X is None else ["exogenous_variable_1"],
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

forecaster = model_class.from_dataset(
    training_dataset,
    log_val_interval=10 if log_val else -1,
)


forecaster.hparams
forecaster_copy = deepcopy(forecaster)

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
pytorch_trainer = Trainer(
    accelerator="cpu",
    max_epochs=500 if (not X_data and model_class is TemporalFusionTransformer) else 1,
    min_epochs=2,
    # callbacks=[early_stop_callback] if log_val else [],
)

start_time = time.time()
print("FIT START")
pytorch_trainer.fit(
    forecaster,
    train_dataloaders=training_data_loader,
    val_dataloaders=validation_data_loader,
)
print("FIT END")
print("TIME: ", time.time() - start_time)

for p, p_c in zip(forecaster.parameters(), forecaster_copy.parameters()):
    print((p - p_c).abs().sum())

best_model_path = pytorch_trainer.checkpoint_callback.best_model_path
best_model = model_class.load_from_checkpoint(best_model_path)
predictions = best_model.predict(
    validation.to_dataloader(**self._validation_to_dataloader_params),
    return_x=True,
    return_index=True,
    return_decoder_lengths=True,
)
# convert pytorch-forecasting predictions to dataframe
output = self._predictions_to_dataframe(predictions, self._max_prediction_length)
