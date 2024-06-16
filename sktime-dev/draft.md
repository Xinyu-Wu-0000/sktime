<details>
<summary>Skipped: ForecastingHorizon with timedelta values is currently experimental and not supported everywhere in `test_predict_time_index`</summary>

```sh
Traceback (most recent call last):
  File "/home/xinyu/Documents/sktime/sktime-dev/py_scripts/check_estimator_TFT.py", line 16, in <module>
    check_estimator(
  File "/home/xinyu/Documents/sktime/./sktime/utils/estimator_checks.py", line 116, in check_estimator
    test_cls_results = test_cls().run_tests(
  File "/home/xinyu/Documents/sktime/./sktime/tests/test_all_estimators.py", line 608, in run_tests
    test_fun(**deepcopy(args))
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/tests/test_all_forecasters.py", line 221, in test_predict_time_index
    pytest.skip(
  File "/home/xinyu/Documents/sktime/env/lib/python3.10/site-packages/_pytest/outcomes.py", line 151, in skip
    raise Skipped(msg=reason, allow_module_level=allow_module_level)
Skipped: ForecastingHorizon with timedelta values is currently experimental and not supported everywhere
```

</details>

<details>
<summary>ValueError: input must be univariate pd.DataFrame, with one column in `test_predict_time_index_with_X`</summary>

```sh
Traceback (most recent call last):
  File "/home/xinyu/Documents/sktime/sktime-dev/py_scripts/check_estimator_TFT.py", line 16, in <module>
    check_estimator(
  File "/home/xinyu/Documents/sktime/./sktime/utils/estimator_checks.py", line 116, in check_estimator
    test_cls_results = test_cls().run_tests(
  File "/home/xinyu/Documents/sktime/./sktime/tests/test_all_estimators.py", line 608, in run_tests
    test_fun(**deepcopy(args))
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/tests/test_all_forecasters.py", line 290, in test_predict_time_index_with_X
    estimator_instance.fit(y_train, X_train, fh=fh)
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/base/_base.py", line 376, in fit
    X_inner, y_inner = self._check_X_y(X=X, y=y)
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/base/_base.py", line 1631, in _check_X_y
    X_inner = convert_to(
  File "/home/xinyu/Documents/sktime/./sktime/datatypes/_convert.py", line 263, in convert_to
    converted_obj = convert(
  File "/home/xinyu/Documents/sktime/./sktime/datatypes/_convert.py", line 182, in convert
    converted_obj = convert_dict[key](obj, store=store)
  File "/home/xinyu/Documents/sktime/./sktime/datatypes/_series/_convert.py", line 89, in convert_MvS_to_UvS_as_Series
    raise ValueError("input must be univariate pd.DataFrame, with one column")
ValueError: input must be univariate pd.DataFrame, with one column
```

</details>

<details>
<summary>ValueError: input must be univariate pd.DataFrame, with one column in `test_fit_predict`</summary>

```sh
Traceback (most recent call last):
  File "/home/xinyu/Documents/sktime/sktime-dev/py_scripts/check_estimator_TFT.py", line 16, in <module>
    check_estimator(
  File "/home/xinyu/Documents/sktime/./sktime/utils/estimator_checks.py", line 116, in check_estimator
    test_cls_results = test_cls().run_tests(
  File "/home/xinyu/Documents/sktime/./sktime/tests/test_all_estimators.py", line 608, in run_tests
    test_fun(**deepcopy(args))
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/tests/test_all_forecasters.py", line 894, in test_fit_predict
    y_pred = estimator_instance.fit_predict(
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/base/_base.py", line 531, in fit_predict
    return self.fit(y=y, X=X, fh=fh).predict(X=X_pred)
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/base/_base.py", line 376, in fit
    X_inner, y_inner = self._check_X_y(X=X, y=y)
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/base/_base.py", line 1631, in _check_X_y
    X_inner = convert_to(
  File "/home/xinyu/Documents/sktime/./sktime/datatypes/_convert.py", line 263, in convert_to
    converted_obj = convert(
  File "/home/xinyu/Documents/sktime/./sktime/datatypes/_convert.py", line 182, in convert
    converted_obj = convert_dict[key](obj, store=store)
  File "/home/xinyu/Documents/sktime/./sktime/datatypes/_series/_convert.py", line 89, in convert_MvS_to_UvS_as_Series
    raise ValueError("input must be univariate pd.DataFrame, with one column")
ValueError: input must be univariate pd.DataFrame, with one column
```

</details>

<details>
<summary>AssertionError: No in sample predict support in `test_predict_time_index_in_sample_full`</summary>

```sh
Traceback (most recent call last):
  File "/home/xinyu/Documents/sktime/sktime-dev/py_scripts/check_estimator_TFT.py", line 16, in <module>
    check_estimator(
  File "/home/xinyu/Documents/sktime/./sktime/utils/estimator_checks.py", line 116, in check_estimator
    test_cls_results = test_cls().run_tests(
  File "/home/xinyu/Documents/sktime/./sktime/tests/test_all_estimators.py", line 608, in run_tests
    test_fun(**deepcopy(args))
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/tests/test_all_forecasters.py", line 323, in test_predict_time_index_in_sample_full
    estimator_instance.fit(y_train, fh=fh)
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/base/_base.py", line 391, in fit
    self._fit(y=y_inner, X=X_inner, fh=fh)
  File "/home/xinyu/Documents/sktime/./sktime/forecasting/base/adapters/_pytorchforecasting.py", line 159, in _fit
    np.min(fh.to_relative(self.cutoff)) > 0
AssertionError: No in sample predict support,         but found fh with in sample index: ForecastingHorizon([-49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36,
       -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22,
       -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10,  -9,  -8,
        -7,  -6,  -5,  -4,  -3,  -2,  -1,   0],
      dtype='int64', is_relative=True)
```

</details>
