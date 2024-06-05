import sys

import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping
from neuralforecast.utils import generate_series
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

sys.path.append(".")
from sktime.datatypes import check_is_mtype, check_is_scitype, check_raise, get_examples
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats

data = generate_series(
    n_series=100,
    min_length=400,
    max_length=400,
    n_temporal_features=3,
    equal_ends=True,
    seed=42,
)
data = data.rename(columns={"unique_id": "series", "y": "value"})
time_idx = list(range(400)) * 100
data["time_idx"] = time_idx
data = data.astype(dict(series=str))
data = data.astype(dict(temporal_0=int, temporal_1=int, temporal_2=int))

max_encoder_length = 60
max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

context_length = max_encoder_length
prediction_length = max_prediction_length

batch_size = 128

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)

ins_index = ["series", "time_idx"]
y_indexs = ["value"]
x_indexs = data.columns.drop(y_indexs + ins_index)
x = data[x_indexs]
y = data[y_indexs]
multi_index = pd.MultiIndex.from_frame(data[ins_index])
x.index = multi_index
y.index = multi_index

check_raise(x, "pd-multiindex")
check_raise(y, "pd-multiindex")

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

model.fit(y, x, fh=fh)
len_levels = len(y.index.names)
_y = y.groupby(level=list(range(len_levels - 1))).apply(
    lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
)
res = model.predict(fh, x, _y)
print(_y)
print(res)
