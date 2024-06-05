python -m pip install -U pip
python -m pip show pytorch-forecasting
python -m pip install .[all_extras_pandas2,dev,binder]
python -m pip install pytorch-forecasting==1.0.0 pandas==2.2.1
time make test_without_datasets
