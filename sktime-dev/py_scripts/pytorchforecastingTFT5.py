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
from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series

data = _make_series()

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

y = data

max_encoder_length = 5
max_prediction_length = 2
# max_prediction_length = 1
# ValueError: 'yerr' must not contain negative values
# pytorch_forecasting/models/base_model.py", line 1051, in plot_prediction
#    ax.errorbar(
# matplotlib/__init__.py", line 1465, in inner
#    return func(ax, *map(sanitize_sequence, args), **kwargs)
batch_size = 128

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)


logger = TensorBoardLogger(
    "sktime-dev/lightning_logs"
)  # logging results to a tensorboard


model = PytorchForecastingTFT(
    model_params={
        # "hidden_size": 30,
        # "attention_head_size": 4,
        # "dropout": 0.1,
        # "hidden_continuous_size": 8,
        # "learning_rate": 1e-3,
        # "log_interval": 10,
        "log_val_interval": 1,
        # "optimizer": "Adam",
        # "reduce_on_plateau_patience": 4,
    },
    dataset_params={
        # "static_reals": ["month", "dayofweek", "dayofyear"],
        # "time_varying_known_reals": [
        #     "Hr [%Hr]",
        #     "RainDur [min]",
        #     "StrGlo [W/m2]",
        #     "T [°C]",
        #     "WD [°]",
        #     "WVs [m/s]",
        #     "WVv [m/s]",
        #     "p [hPa]",
        # ],
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
    },
    train_to_dataloader_params={
        # "batch_size": batch_size,
        # "num_workers": 0,
    },
    validation_to_dataloader_params={
        # "batch_size": batch_size,
        # "num_workers": 0,
    },
    trainer_params={
        "max_epochs": 1,
        # "accelerator": "cpu",
        # "enable_model_summary": True,
        # "gradient_clip_val": 0.1,
        "limit_train_batches": 10,  # coment in for training, running valiation every 30 batches
        # # "fast_dev_run": True,  # comment in to check that networkor dataset has no serious bugs
        # "callbacks": [early_stop_callback],
        # "enable_checkpointing": True,
        "logger": logger,
    },
)


fh = ForecastingHorizon(np.arange(1, max_prediction_length + 1), is_relative=True)
print(fh.to_relative()[-1])

model.fit(y=y, fh=fh)
len_levels = len(y.index.names)
_y = y.iloc[:-max_prediction_length]
res = model.predict(fh, y=_y)
print(_y)
print(res)
