from sktime.transformations.series import kalman_filter as kf
from sktime.utils.estimator_checks import check_estimator

check_estimator(kf.KalmanFilterTransformerFP)
# All tests PASSED!
check_estimator(kf.KalmanFilterTransformerPK, raise_exceptions=True)
# Traceback (most recent call last):
#   File "/home/xinyu/Documents/sktime/sktime-dev/temp/test_KalmanFilterTransformer.py", line 7, in <module>
#     check_estimator(kf.KalmanFilterTransformerPK, raise_exceptions=True)
#   File "/home/xinyu/Documents/sktime/sktime/utils/estimator_checks.py", line 116, in check_estimator
#     test_cls_results = test_cls().run_tests(
#                        ^^^^^^^^^^^^^^^^^^^^^
#   File "/home/xinyu/Documents/sktime/sktime/tests/test_all_estimators.py", line 605, in run_tests
#     test_fun(**deepcopy(args))
#   File "/home/xinyu/Documents/sktime/sktime/tests/test_all_estimators.py", line 1230, in test_fit_does_not_overwrite_hyper_params
#     fitted_est = scenario.run(estimator_instance, method_sequence=["fit"])
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/xinyu/Documents/sktime/sktime/utils/_testing/scenarios.py", line 180, in run
#     res = getattr(obj, methodname)(**args)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/xinyu/Documents/sktime/sktime/transformations/base.py", line 498, in fit
#     self._fit(X=X_inner, y=y_inner)
#   File "/home/xinyu/Documents/sktime/sktime/transformations/series/kalman_filter.py", line 640, in _fit
#     ) = self._em(X=X, measurement_dim=measurement_dim, state_dim=self.state_dim)
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/xinyu/Documents/sktime/sktime/transformations/series/kalman_filter.py", line 813, in _em
#     kf = KalmanFilter(
#          ^^^^^^^^^^^^^
#   File "/home/xinyu/Documents/sktime/env/lib/python3.12/site-packages/pykalman/standard.py", line 1029, in __init__
#     n_dim_obs = _determine_dimensionality(
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/xinyu/Documents/sktime/env/lib/python3.12/site-packages/pykalman/standard.py", line 75, in _determine_dimensionality
#     raise ValueError(
# ValueError: The shape of all parameters is not consistent.  Please re-check their values.
