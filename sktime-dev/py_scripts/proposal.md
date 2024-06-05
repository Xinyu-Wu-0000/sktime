
## Basic Information

Hi! I'm Xinyu Wu, a Master's student in Control Theory at the Harbin Institute of Technology, Shenzhen campus ([HITSZ](http://global.hitsz.edu.cn/info/1047/1316.htm)) in China (UTC+8).I graduated with a Bachelor's degree in Automation from HITSZ last year, and I'm currently pursuing my MPhil from September 2023 to April 2025. My Github account is [Xinyu-Wu-0000](https://github.com/Xinyu-Wu-0000). My email address is xinyu.wu.hitsz@gmail.com. I have also joined the sktime discord group.

I'm fluent in Python, C/C++, and Julia, with experience using Matlab and [Simulink](https://www.mathworks.com/help/simulink/). I also have a basic understanding of HTML, CSS, and Javascript. For my Bachelor's Degree Final Project, I developed a multi-agent reinforcement learning project in Python using PyTorch to guide multiple robots to targets while avoiding collisions. The simulation environment and observation, reward calculations were implemented in C++ using [MuJoCo](https://mujoco.org/) as the physics engine. The C++ code is interfaced to python with PyBind11. My passion for technology covers many topics. I have experience working with Docker,  [ROS](https://www.ros.org/) (Robot Operating System), [OpenCV](https://opencv.org/) (computer vision library), [Vercel](https://vercel.com/) (deployment platform), and [microcontrollers](https://en.wikipedia.org/wiki/Microcontroller)...etc.

I have opened two Pull Request to sktime:

[ENH][BUG] Second test parameter set for shapeDTW [#6093](https://github.com/sktime/sktime/pull/6093)

[ENH] Second test parameter set for Kalman Filter [#6095](https://github.com/sktime/sktime/pull/6095)
## Project Introduction

I propose merging two existing GSoC projects: "scaling, backends, foundation models - `polars`, `pytorch`, `huggingface`" and "global forecasting and reduction forecasting." This combined project would focus on interfacing `darts` and `pytorch-forecast` with a global forecasting API within sktime. There are several compelling reasons to combine these projects:
1. **Deeply Coupled Projects:** Based on discussions in "[ENH] Support Global Time Series Forecasting [#4651](https://github.com/sktime/sktime/issues/4651)," these projects appear to be inherently interconnected.
2. **Phased Approach to `predict` Extension:** Extending the `predict` functionality will impact numerous existing estimators. This requires work across API design, coding, testing, documentation, and potentially even mathematical design (given the absence of a global ARIMA method as discussed in [#5021](https://github.com/sktime/sktime/issues/5021). A less risky and more manageable approach would be to initially test this extension with new estimators by interfacing `darts` or `pytorch-forecast`.
3. **Future-Proofing for Deep Learning Models:** Global forecasting relies heavily on neural networks and deep learning, with this trend likely to continue. By starting with deep learning models in `darts` and `pytorch-forecast`, we can identify potential design issues in the global forecasting API that might hinder the integration of future deep learning-based models.
4. **Saving Future Work**: sktime already has a `NeuralForecast` interface without a global forecasting API. Addressing this API gap is inevitable. By interfacing `darts` and `pytorch-forecast` with the global forecasting API from the outset, we can significantly reduce future development effort since most models interfaced through these libraries will require the API functionality.
# Project Implementation

Basically, the interfacing of `darts` and `pytorch-forecasting` would be like  `_NeuralForecastAdapter` and `_GeneralisedStatsForecastAdapter`.

### Base Class

The inheritance chain would be `BaseEstimator`->`BaseForecaster`->`_DartsAdapter`/`_PytorchForecastingAdapter`->`SpecificForecasters`. To enable global forecasting functionality, we need to extend the global forecasting API within `BaseForecaster`. However, modifying `BaseForecaster` directly can be disruptive as it's the base class for many existing forecasters. A possible phased approach could be that we temporally create a new `BaseForecaster` with global forecasting API named `GlobalBaseForecaster`, then we create the adapter classes `_DartsAdapter`/`_PytorchForecastingAdapter` inheriting from the new `GlobalBaseForecaster`.  After testing and validating the design, we can merge `GlobalBaseForecaster` back into `BaseForecaster`. This phased approach allows for controlled introduction of the global forecasting API while minimizing the impact on existing code.

We can check the tag "capability:global_forecast" in `BaseForecaster` and raise error if `y` is passed to `predict` of a non-global forecaster. We can also check the tag "capability:global_only" in `BaseForecaster` and raise error if `y` is not passed to `predict` of a global-only forecaster. Of cause, we can also use "capability:local_forecast" instead of "capability:global_only" if we would like to add it to all existing local forecasters. For forecaster with both local and global forecast ability, perform a local forecast if no `y` passed, a global forecast with `y` passed.

### Adapter Class

The most important function of adapter classes would be `_fit` and `_predict`.

In `_fit`, we need to implement data type converting from sktime data type to `darts` data type ([Timeseries](https://unit8co.github.io/darts/generated_api/darts.timeseries.html)) or `pytorch-forecasting `data type ([TimeSeriesDataSet](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html#pytorch_forecasting.data.timeseries.TimeSeriesDataSet "pytorch_forecasting.data.timeseries.TimeSeriesDataSet")). Both Timeseries and TimeSeriesDataSet can be created from panda data type with build in support. After data type converting we can fit model by `fit` function in dart models and `from_dataset` function in `pytorch-forecasting` models. For `pytorch-forecasting`, model instantiation and fitting are combined, but for darts, the two steps are separate.

In `_predict`, we also need to implement data type converting first. Then we can call `predict` for both `darts` models and `pytorch-forecasting` models to get the result. Then we need to implement data converting from `darts` and `pytorch-forecasting` data type back to sktime data type. `Timeseries.pd_dataframe` and `Timeseries.pd_series` can be used to convert `darts` data type to pandas data type. To convert `pytorch-forecasting` `TimeSeriesDataSet`, we might need to write our own converting function.

Model instantiation can be designed just like `_NeuralForecastAdapter` and `_GeneralisedStatsForecastAdapter`: implement `_instantiate_model` in adapter, `algorithm_class` and `algorithm_parameters` in specific forecasters. We can also initialize Some shared parameters in `__init__` of adapter class such as `log_interval`, `learning_rate`, `weight_decay`, `output_transformer` in `pytorch-forecasting` adapter. But since `darts` include both statistical models and deep learning models, no parameters can be shared across different models (I can't find one). If we only focus on torch models in `darts`,  some shared parameters could be `input_chunk_length`, `output_chunk_length`, `output_chunk_shift`.

### Specific Forecasters

In each specific forecaster, we need to initialize forecaster specific parameters and `algorithm_parameters` method to return the `dict` of parameters. A `algorithm_class` method is also needed to return the underlying model class. With `algorithm_class` and `algorithm_parameters`, the adapter class should have all the blocks to instantiate the model. `_tags` and `get_test_params` should also be implemented in specific forecasters.

For darts, I would choose 4 global forecasters to implement: [`RandomForest`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html#darts.models.forecasting.random_forest.RandomForest "darts.models.forecasting.random_forest.RandomForest"), [`XGBModel`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html#darts.models.forecasting.xgboost.XGBModel "darts.models.forecasting.xgboost.XGBModel"), [`DLinearModel`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.dlinear.html#darts.models.forecasting.dlinear.DLinearModel "darts.models.forecasting.dlinear.DLinearModel"), [`NLinearModel`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nlinear.html#darts.models.forecasting.nlinear.NLinearModel "darts.models.forecasting.nlinear.NLinearModel").
For pytorch forecasting, I would also choose 4 global forecasters to implement: [`NBeats`](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nbeats.NBeats.html#pytorch_forecasting.models.nbeats.NBeats "pytorch_forecasting.models.nbeats.NBeats"), [`NHiTS`](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nhits.NHiTS.html#pytorch_forecasting.models.nhits.NHiTS "pytorch_forecasting.models.nhits.NHiTS"), [`DeepAR`](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.deepar.DeepAR.html#pytorch_forecasting.models.deepar.DeepAR "pytorch_forecasting.models.deepar.DeepAR"), [`TemporalFusionTransformer`](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html#pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer "pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer").
Of course, it's unlikely that I can implement 2 adapters and 8 forecasters in the limited time period. I expect to implement one adapters and 4 forecasters during GSoC.

For `pytorch-forecasting`, I simply choose 4 most highlighted forecasters from limited choices, 2 MLP-based models, one RNN-based model and one transformer-based model . For `darts`, I choose 4 models not included in `pytorch-forecasting`, 2 commonly used regression models and 2 MLP-based models.

## Why Sktime

1. Sktime is an active community, well organised, openly governed and community-driven, making it a great place to begin my open-source journey.
2. I have great interest on Finance and Economics. I even took the Principles of Economics class as an auditor last semester.
3. Series prediction is deeply coupled with my academic research. Predicting the move of other robots or pedestrians is series prediction. Path planning for robots or robot arms can be done by predict the best trajectory. Choosing a good path in [sampling-based Path Planning Algorithms](https://arxiv.org/abs/2304.14839) is series regression.
### Other Thoughts

About `ray` :
StatsForecast is designed with out-of-the-box compatibility with Spark, Dask, and Ray. Auto models in NeuralForecast can choose backend `ray` or `optuna`. All deep learning models in darts are implemented using PyTorch Lightning. Pytorch forecasting uses PyTorch Lightning and optuna. It would be really complicated if we implement our own and interface those models at the same time.

About `polars`:
Using `polars` as a new `mtype` to have handle huge datasets is a promising idea, but we also need to extend the API of `scitype` and our training mechanism. It would be a big project with a lot of design.

About reduction forecasting and `huggingface`:
I do not have enough knowledge of reduction forecasting, and since [benHeid](https://github.com/benHeid) is working on [ENH] Huggingface Forecaster [#5796](https://github.com/sktime/sktime/pull/5796), I haven't investigate those two topics yet, but I am glad to spend time on them if needed.

## Timeline

### Pre GSoC
- Study of those forecasting models need to implement.
- Learn necessary technics like exogenous time series mapping, pytest.
### Community Bonding
- Get more acquainted with the code base of sktime, darts and pytorch forecasting.
- Decide which to implement first, darts or pytorch forecasting
- Discuss the details of the implementation with mentors.
### Week 1
- create the draft of the adapter class
### Week 2
- create the draft of the first forecaster model
### Week 3
- write the test code for the first forecaster model
### Week 4
- finish the test and debug, write the document
### Week 5~11
- implement the other 3 forecaster models
- coding, testing, documenting
- 2.3 weeks for each forecaster in average
## Week 12
- Spare week to do more fixes and testes.
