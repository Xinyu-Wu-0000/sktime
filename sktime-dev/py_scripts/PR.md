#### Reference Issues/PRs
Related: https://github.com/sktime/sktime/issues/4651, https://github.com/sktime/sktime/issues/4923

#### What does this implement/fix? Explain your changes.
A pytorch forecasting adapter with Global Forecasting API with several algorithms to validate the design.

#### Does your contribution introduce a new dependency? If yes, which one?
Soft dependencies: pytorch-forecasting, pytorch-lightning

I am trying to create a pytorch-forecasting adapter with Global Forecasting API, but I am not sure about some design, so I create this draft PR for discusstion.

I added a new base class `GlobalBaseForecaster` to avoid effecting other forecasters and minimize the changes and possible errors in testes. According to discussion in https://github.com/sktime/sktime/issues/4651, global forecasting api will be managing
 via tags only, but maybe we can do it in a phased approach. If we still want it be tags only, we can merge `GlobalBaseForecaster` back to `BaseForecaster` after we validate the design.
