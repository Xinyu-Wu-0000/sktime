# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for pytorch-forecasting models."""
import abc
import functools
import typing
from typing import Any, Dict, List, Optional

import pandas
from pandas.api.types import is_numeric_dtype

from sktime.forecasting.base import BaseGlobalForecaster, ForecastingHorizon

__all__ = ["_PytorchForecastingAdapter"]
__author__ = ["XinyuWu"]


class _PytorchForecastingAdapter(BaseGlobalForecaster):
    """Base adapter class for pytorch-forecasting models."""

    _tags = {
        # packaging info
        # --------------
        "authors": ["XinyuWu"],
        "maintainers": ["XinyuWu"],
        "python_version": ">=3.8",
        "python_dependencies": ["pytorch_forecasting"],
        # estimator type
        # --------------
        "y_inner_mtype": ["pd-multiindex", "pd_multiindex_hier", "pd.Series"],
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "univariate",
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "handles-missing-data": False,
        "capability:insample": False,
    }

    def __init__(
        self: "_PytorchForecastingAdapter",
        model_params: Optional[Dict[str, Any]] = None,
        allowed_encoder_known_variable_names: Optional[List[str]] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        train_to_dataloader_params: Optional[Dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[Dict[str, Any]] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model_params = model_params
        self.allowed_encoder_known_variable_names = allowed_encoder_known_variable_names
        self.dataset_params = dataset_params
        self.trainer_params = trainer_params
        self.train_to_dataloader_params = train_to_dataloader_params
        self.validation_to_dataloader_params = validation_to_dataloader_params

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_class(self: "_PytorchForecastingAdapter"):
        """Import underlying pytorch-forecasting algorithm class."""

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_parameters(self: "_PytorchForecastingAdapter") -> dict:
        """Keyword parameters for the underlying pytorch-forecasting algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class

        """

    def _instantiate_model(self: "_PytorchForecastingAdapter", data):
        """Instantiate the model."""
        self._model_params = _none_check(self.model_params, {})
        algorithm_instance = self.algorithm_class.from_dataset(
            data,
            self.allowed_encoder_known_variable_names,
            **self.algorithm_parameters,
            **self._model_params,
        )
        self._trainer_params = _none_check(self.trainer_params, {})
        import lightning.pytorch as pl

        traner_instance = pl.Trainer(**self._trainer_params)
        return algorithm_instance, traner_instance

    def _fit(
        self: "_PytorchForecastingAdapter",
        y: pandas.DataFrame,
        X: typing.Optional[pandas.DataFrame],
        fh: ForecastingHorizon,
    ) -> "_PytorchForecastingAdapter":
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to have a single column/variable
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to have at least one column/variable
            Exogeneous time series to fit to.

        Returns
        -------
        self : _PytorchForecastingAdapter
            reference to self

        Raises
        ------
        ValueError
            When ``freq="auto"`` and cannot be interpreted from ``ForecastingHorizon``
        """
        self._dataset_params = _none_check(self.dataset_params, {})
        self._max_prediction_length = fh.to_relative()[-1]
        if isinstance(y, pandas.Series):
            _y = y.to_frame()
        else:
            _y = y
        # store the target column name
        self.y_name = _y.columns[-1]
        # convert data to pytorch-forecasting datasets
        training, validation = self._Xy_to_dataset(
            X, _y, self._dataset_params, self._max_prediction_length
        )
        self._forecaster, self._trainer = self._instantiate_model(training)
        self._train_to_dataloader_params = {"train": True}
        self._train_to_dataloader_params.update(
            _none_check(self.train_to_dataloader_params, {})
        )
        self._validation_to_dataloader_params = {"train": False}
        self._validation_to_dataloader_params.update(
            _none_check(self.validation_to_dataloader_params, {})
        )
        self._trainer.fit(
            self._forecaster,
            train_dataloaders=training.to_dataloader(
                **self._train_to_dataloader_params
            ),
            val_dataloaders=validation.to_dataloader(
                **self._validation_to_dataloader_params
            ),
        )
        return self

    def _predict(
        self: "_PytorchForecastingAdapter",
        fh: typing.Optional[ForecastingHorizon],
        X: typing.Optional[pandas.DataFrame],
        y: typing.Optional[pandas.DataFrame],
    ) -> pandas.Series:
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        y : sktime time series object, optional (default=None)
            Historical values of the time series that should be predicted.

        Returns
        -------
        y_pred : sktime time series object
            guaranteed to have a single column/variable
            Point predictions
        """
        # convert data to pytorch-forecasting datasets
        training, validation = self._Xy_to_dataset(
            X, y, self._dataset_params, self._max_prediction_length
        )
        # load model from checkpoint
        best_model_path = self._trainer.checkpoint_callback.best_model_path
        best_model = self.algorithm_class.load_from_checkpoint(best_model_path)
        predictions = best_model.predict(
            validation.to_dataloader(**self._validation_to_dataloader_params),
            return_x=True,
            return_index=True,
            return_decoder_lengths=True,
        )
        # convert pytorch-forecasting predictions to dataframe
        output = self._predictions_to_dataframe(
            predictions, self._max_prediction_length
        )

        return output

    def _Xy_to_dataset(
        self,
        X: pandas.DataFrame,
        y: pandas.DataFrame,
        dataset_params: Dict[str, Any],
        max_prediction_length,
    ):
        from pytorch_forecasting.data import TimeSeriesDataSet

        # X, y must have same index
        assert (X.index == y.index).all()
        # warning! X will be modified
        time_varying_known_reals = [
            c for c in X.columns if is_numeric_dtype(X[c].dtype)
        ]
        data = X.join(y, on=X.index.names)
        index_names = data.index.names
        self._time_idx_name = index_names[-1]
        index_lens = index_names.__len__()
        # add int time_idx as pytorch-forecasting requires
        time_idx = data.groupby(by=index_names[0:-1]).cumcount().to_frame()
        time_idx.rename(columns={0: "_auto_time_idx"}, inplace=True)
        data = data.join(time_idx, on=data.index.names)
        # reset multi index to normal columns
        data = data.reset_index(level=list(range(index_lens)))
        training_cutoff = data["_auto_time_idx"].max() - max_prediction_length
        # save origin time idx for prediction
        self._origin_time_idx = data[index_names + ["_auto_time_idx"]][
            data["_auto_time_idx"] > training_cutoff
        ]
        # infer time_idx column, target column and instances from data
        _dataset_params = {
            "data": data[data["_auto_time_idx"] <= training_cutoff],
            "time_idx": "_auto_time_idx",
            "target": data.columns[-2],
            "group_ids": index_names[0:-1],
            "time_varying_known_reals": time_varying_known_reals,
            "time_varying_unknown_reals": [data.columns[-2]],
        }
        _dataset_params.update(dataset_params)
        # overwrite max_prediction_length
        _dataset_params["max_prediction_length"] = int(max_prediction_length)
        training = TimeSeriesDataSet(**_dataset_params)
        validation = TimeSeriesDataSet.from_dataset(
            training, data, predict=True, stop_randomization=True
        )
        return training, validation

    def _predictions_to_dataframe(self, predictions, max_prediction_length):
        # output is the predictions
        output = predictions.output.cpu().numpy()
        # index will be combined with output
        index = predictions.index
        # in pytorch-forecasting predictions, the first index is the time_idx
        columns_names = index.columns.to_list()
        time_idx = columns_names.pop(0)
        # make time_idx the last index
        columns_names.append(time_idx)
        # in pytorch-forecasting predictions,
        # the index only contains the start timepoint.
        data = index.loc[index.index.repeat(max_prediction_length)].reset_index(
            drop=True
        )
        # make time_idx the last index
        data = data.reindex(columns=columns_names)
        # add the target column at the end
        data[self.y_name] = output.flatten()
        # correct the time_idx after repeating
        # assume the time_idx column is continuous integers
        for i in range(output.shape[0]):
            start_idx = i * max_prediction_length
            start_time = data.loc[start_idx, time_idx]
            data.loc[
                start_idx : start_idx + max_prediction_length - 1, time_idx
            ] = list(range(start_time, start_time + max_prediction_length))

        # set the instance columns to multi index
        data.set_index(columns_names, inplace=True)
        self._origin_time_idx.set_index(columns_names, inplace=True)
        # add origin time_idx column to data
        data = data.join(self._origin_time_idx, on=columns_names)
        # drop _auto_time_idx column
        data.reset_index(level=list(range(len(columns_names))), inplace=True)
        data.drop("_auto_time_idx", axis=1, inplace=True)
        columns_names.remove("_auto_time_idx")
        # reindex to origin multiindex
        data.set_index(
            columns_names + [self._time_idx_name],
            inplace=True,
        )

        return data


def _none_check(value, default):
    return value if value is not None else default
