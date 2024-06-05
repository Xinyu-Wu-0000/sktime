import sys

sys.path.append(".")
import numpy as np
import pandas as pd
from darts.datasets import ElectricityConsumptionZurichDataset
from lightning.pytorch.callbacks import EarlyStopping

from sktime.datatypes import check_raise
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats

data = ElectricityConsumptionZurichDataset().load().pd_dataframe()
data.reset_index([0], inplace=True)
data["month"] = data["Timestamp"].dt.month
data["dayofweek"] = data["Timestamp"].dt.dayofweek
data["dayofyear"] = data["Timestamp"].dt.dayofyear
data["days"] = (data["Timestamp"] - data["Timestamp"][0]).dt.days
data["time_idx"] = (
    (data["Timestamp"].dt.hour * 60 + data["Timestamp"].dt.minute) / 15 + 1
).astype(np.int64)
data.drop(data.tail(1).index, inplace=True)

ins_index = ["days", "time_idx"]
multi_index = pd.MultiIndex.from_frame(data[ins_index])
y_indexs = ["Value_NE5", "Value_NE7"]
x_indexs = data.columns.drop(y_indexs + ins_index)
x = data[x_indexs]
y = data[y_indexs]
x.index = multi_index
y.index = multi_index

y1 = y["Value_NE5"].to_frame()
y2 = y["Value_NE7"].to_frame()
check_raise(x, "pd-multiindex")
check_raise(y1, "pd-multiindex")
check_raise(y2, "pd-multiindex")

max_encoder_length = 64
max_prediction_length = 10

training_cutoff = data["time_idx"].max() - max_prediction_length

batch_size = 128

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
from lightning.pytorch.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    "sktime-dev/lightning_logs"
)  # logging results to a tensorboard


model = PytorchForecastingNBeats(
    model_params={
        "num_blocks": [5, 5],
        "num_block_layers": [5, 5],
        "log_interval": 10,
        "backcast_loss_ratio": 1.0,
    },
    dataset_params={
        "max_encoder_length": max_encoder_length,
        # "min_encoder_length": 5,
    },
    train_to_dataloader_params={
        "batch_size": batch_size,
        "num_workers": 0,
    },
    validation_to_dataloader_params={
        "batch_size": batch_size,
        "num_workers": 0,
    },
    trainer_params={
        "max_epochs": 1,
        # "accelerator": "cpu",
        # "enable_model_summary": True,
        # "gradient_clip_val": 0.1,
        "limit_train_batches": 10,  # coment in for training, running valiation every 30 batches
        # # "fast_dev_run": True,  # comment in to check that networkor dataset has no serious bugs
        "callbacks": [early_stop_callback],
        # "enable_checkpointing": True,
        "logger": logger,
    },
)


fh = ForecastingHorizon(np.arange(1, max_prediction_length + 1), is_relative=True)
print(fh.to_relative()[-1])

model.fit(y1, x, fh=fh)
len_levels = len(y1.index.names)
_y = y1.groupby(level=list(range(len_levels - 1))).apply(
    lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
)
res = model.predict(fh, x, _y)
print(_y)
print(res)
