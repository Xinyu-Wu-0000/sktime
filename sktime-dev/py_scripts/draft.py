import numpy as np
import pandas as pd
from lightning.pytorch import Trainer
from pytorch_forecasting import (
    DeepAR,
    NBeats,
    NHiTS,
    QuantileLoss,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)

model_class = DeepAR
max_prediction_length = 2

n_timeseries = 100
time_points = 100
data = pd.DataFrame(
    data={
        "target": np.random.rand(time_points * n_timeseries),
        "time_varying_known_real_1": np.random.rand(time_points * n_timeseries),
        "time_idx": np.tile(np.arange(time_points), n_timeseries),
        "group_id": np.repeat(np.arange(n_timeseries), time_points),
    }
)

print(data)

training_cutoff = data["time_idx"].max() - max_prediction_length
training_dataset = TimeSeriesDataSet(
    data=data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="target",
    group_ids=["group_id"],
    time_varying_unknown_reals=["target"],
    time_varying_known_reals=(
        ["time_varying_known_real_1"] if model_class != NBeats else []
    ),
    max_prediction_length=max_prediction_length,
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    data,
    stop_randomization=True,
    predict=True,
)

training_data_loader = training_dataset.to_dataloader(train=True)

validation_data_loader = validation_dataset.to_dataloader(train=False)

forecaster = model_class.from_dataset(
    training_dataset,
    log_val_interval=-1,
    output_size=7,
    loss=QuantileLoss(),
)

pytorch_trainer = Trainer(
    accelerator="cpu",
    max_epochs=5,
    min_epochs=2,
    limit_train_batches=10,
)

pytorch_trainer.fit(
    forecaster,
    train_dataloaders=training_data_loader,
    val_dataloaders=validation_data_loader,
)

best_model_path = pytorch_trainer.checkpoint_callback.best_model_path
best_model = model_class.load_from_checkpoint(best_model_path)
predictions = best_model.predict(
    validation_data_loader,
    mode="raw",
    # return_x=True,
    # return_index=True,
    # return_decoder_lengths=True,
)
print(predictions)
print(predictions.prediction.size())
print(predictions.output.shape)
