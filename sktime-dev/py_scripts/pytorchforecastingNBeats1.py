import sys

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import QuantileLoss

sys.path.append(".")
from sktime.datatypes import check_raise
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats

data = get_stallion_data()

# add time index
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# add additional features
data["month"] = data.date.dt.month.astype(str).astype(
    "category"
)  # categories have be strings
data["log_volume"] = np.log(data.volume + 1e-8)
data["avg_volume_by_sku"] = data.groupby(
    ["time_idx", "sku"], observed=True
).volume.transform("mean")
data["avg_volume_by_agency"] = data.groupby(
    ["time_idx", "agency"], observed=True
).volume.transform("mean")

# we want to encode special days as one variable and thus need to first reverse one-hot encoding
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
data[special_days] = (
    data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
)

max_prediction_length = 6
max_encoder_length = 24
batch_size = 128  # set this between 32 to 128

# configure network and trainer
pl.seed_everything(42)
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor()  # log the learning rate
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
    model_path="sktime-dev/lightning_logs/lightning_logs/version_14/checkpoints/epoch=0-step=10.ckpt",
)

# get the x, y from data
ins_index = ["agency", "sku", "time_idx"]
y_indexs = ["volume"]
x_indexs = data.columns.drop(y_indexs + ins_index)
x = data[x_indexs]
y = data[y_indexs]
multi_index = pd.MultiIndex.from_frame(data[ins_index])
x.index = multi_index
y.index = multi_index


check_raise(x, "pd_multiindex_hier")
check_raise(y, "pd_multiindex_hier")

fh = ForecastingHorizon(
    np.arange(1, max_prediction_length + 1, dtype=np.int64), is_relative=True
)

model.fit(y, x, fh)
len_levels = len(y.index.names)
_y = y.groupby(level=list(range(len_levels - 1))).apply(
    lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
)
res = model.predict(fh, x, _y)
print(_y)
print(res)
