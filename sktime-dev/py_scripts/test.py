import sys

sys.path.append(".")

from sktime.forecasting.arima import ARIMA
from sktime.forecasting.var import VAR
from sktime.utils.estimator_checks import check_estimator

check_estimator(
    ARIMA,
    raise_exceptions=True,
)

# check_estimator(
#     VAR,
#     # fixtures_to_run=["test_predict_time_index[VAR-1-fh=-3-datetime-datetime-False]"],
#     raise_exceptions=True,
# )
