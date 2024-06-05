# import sys

# sys.path.append(".")

from sktime.forecasting.pytorchforecasting import (
    PytorchForecastingNBeats,
    PytorchForecastingTFT,
)
from sktime.utils.estimator_checks import check_estimator

check_estimator(PytorchForecastingNBeats)
check_estimator(PytorchForecastingTFT)
