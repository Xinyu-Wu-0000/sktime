# sktime/utils/estimator_checks.py col 129
# msg = ["FAILED: " + f"'{x}'," for x in msg]
import sys

sys.path.append(".")
from inspect import signature

from sktime.forecasting.pytorchforecasting import (
    PytorchForecastingNBeats,
    PytorchForecastingTFT,
)
from sktime.utils.estimator_checks import check_estimator

model_to_check = PytorchForecastingTFT

# check_estimator(
#     model_to_check,
#     # tests_to_exclude=[
#     #     "test_global_forecasting_tag",
#     #     "test_pridect_signature",
#     #     "test_global_fit_predict_insample",
#     # ],
#     # raise_exceptions=True,
# )

# check_estimator(
#     model_to_check,
#     tests_to_run=[
#         "test_global_forecasting_tag",
#         "test_pridect_signature",
#         "test_global_forecasting_multiindex_hier",
#         "test_global_forecasting_multiindex",
#         "test_global_forecasting_series",
#         "test_global_forecasting_no_X",
#     ],
#     raise_exceptions=True,
# )

check_estimator(
    model_to_check,
    tests_to_run=[
        "test_global_forecasting_series",
    ],
    raise_exceptions=True,
)

# check_estimator(
#     model_to_check,
#     fixtures_to_run=[
#         "test_fit_predict[PytorchForecastingTFT-0-y:1cols]",
#     ],
#     raise_exceptions=True,
# )
