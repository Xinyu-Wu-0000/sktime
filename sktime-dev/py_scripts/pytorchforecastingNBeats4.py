import sys
import time

import numpy as np
import pandas as pd
from darts.datasets import ElectricityConsumptionZurichDataset
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.append(".")
from sktime.datatypes import check_raise
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats
from sktime.utils._testing.hierarchical import _make_hierarchical

data = _make_hierarchical(
    hierarchy_levels=(5, 100), max_timepoints=100, min_timepoints=100, n_columns=3
)

# index_names = list(data.index.names)
# index_lens = index_names.__len__()
# # reset multi index to normal columns\
# # time_idx_name = index_names.pop()
# # print(data.index.dtypes.iloc[-1])
# time_idx = data.groupby(by=index_names[0:-1]).cumcount().to_frame()
# data = data.join(time_idx, on=data.index.names)
# data = data.rename(columns={0: "_auto_time_idx"})
# print(data.index)
# data = data.reset_index(level=list(range(index_lens)))
# print(data)
# exit()

x = data[["c0", "c1"]]
y = data["c2"].to_frame()
# y.rename(columns={"c2": 0}, inplace=True)
check_raise(x, "pd_multiindex_hier")
check_raise(y, "pd_multiindex_hier")

max_encoder_length = 64
max_prediction_length = 10

batch_size = 128

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)


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
