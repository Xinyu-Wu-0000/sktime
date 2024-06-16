# sktime/utils/estimator_checks.py col 129
# msg = ["FAILED: " + f"'{x}'," for x in msg]
import sys

sys.path.append(".")
from inspect import signature

from sktime.forecasting.pytorchforecasting import PytorchForecastingDeepAR
from sktime.utils.estimator_checks import check_estimator

model_to_check = PytorchForecastingDeepAR

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
#         "test_predict_time_index_in_sample_full", # in sample
#         "test_hierarchical_with_exogeneous", # no y in predict
#         "test_predict_time_index_with_X", # no y in predict
#         "test_score", # no X and y in predict
#         "test_predict_time_index", # no X and y in predict
#         "test_fit_predict", # fit_predict no y in predict
#         "test_update_predict_predicted_index", # update_predict no y in predict
#         "test_update_predict_single", # update_predict_single no y in predict
#     ],
#     raise_exceptions=True,
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
    fixtures_to_run=["test_score[PytorchForecastingDeepAR-0-y:2cols-fh=1]"],
    raise_exceptions=True,
)
