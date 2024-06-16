import sys

sys.path.append(".")

# import packages
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.pytorchforecasting import PytorchForecastingNHiTS
from sktime.utils._testing.hierarchical import _make_hierarchical

# generate random data
data = _make_hierarchical(
    hierarchy_levels=(5, 200), max_timepoints=50, min_timepoints=50, n_columns=3
)
# define forecast horizon
max_prediction_length = 5
fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)
# split X, y data for train and test
l1 = data.index.get_level_values(1).map(lambda x: int(x[3:]))
X_train = data.loc[l1 < 190, ["c0", "c1"]]
y_train = data.loc[l1 < 190, "c2"].to_frame()
X_test = data.loc[l1 >= 180, ["c0", "c1"]]
y_test = data.loc[l1 >= 180, "c2"].to_frame()
len_levels = len(y_test.index.names)
y_test = y_test.groupby(level=list(range(len_levels - 1))).apply(
    lambda x: x.droplevel(list(range(len_levels - 1))).iloc[:-max_prediction_length]
)
# define the model
model = PytorchForecastingNHiTS(
    trainer_params={
        "max_epochs": 5,  # for quick test
        "limit_train_batches": 10,  # for quick test
    },
)
# fit and predict
model.fit(y=y_train, X=X_train, fh=fh)
y_pred = model.predict(fh, X=X_test, y=y_test)
print(y_test)
print(y_pred)
