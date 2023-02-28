
from __future__ import annotations
import torch
import math
from botorch.cross_validation import gen_loo_cv_folds
device = torch.device("cpu")
dtype = torch.float
torch.manual_seed(3)
from botorch.cross_validation import batch_cross_validation
from botorch.models import FixedNoiseGP, SingleTaskGP, SaasFullyBayesianSingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from icecream import ic
from botorch.utils.transforms import normalize, standardize
import numpy as np
from scipy.stats import fisher_exact, norm, pearsonr, spearmanr


from botorch.models.transforms import Standardize
from botorch import fit_fully_bayesian_model_nuts



from typing import Any, Dict, NamedTuple, Optional, Type

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim.utils import _filter_kwargs
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor

from botorch.models.transforms.outcome import Standardize as Standardize_Outcome


from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)

class CVFolds(NamedTuple):
    train_X: Tensor
    test_X: Tensor
    train_Y: Tensor
    test_Y: Tensor
    train_Yvar: Optional[Tensor] = None
    test_Yvar: Optional[Tensor] = None


class CVResults(NamedTuple):
    model: GPyTorchModel
    posterior: GPyTorchPosterior
    observed_Y: Tensor
    observed_Yvar: Optional[Tensor] = None



def _mape(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_pred - y_obs) / y_obs)))

def _correlation_coefficient(
    y_obs: np.ndarray, y_pred: np.ndarray,
) -> float:
    with np.errstate(invalid="ignore"):
        rho, _ = pearsonr(y_pred, y_obs)
    return float(rho)

def _batch_cross_validation(
    model_cls: Type[GPyTorchModel],
    mll_cls: Type[MarginalLogLikelihood],
    cv_folds: CVFolds,
    fit_args: Optional[Dict[str, Any]] = None,
    observation_noise: bool = False,
) -> CVResults:
    r"""Perform cross validation by using gpytorch batch mode.

    Args:
        model_cls: A GPyTorchModel class. This class must initialize the likelihood
            internally. Note: Multi-task GPs are not currently supported.
        mll_cls: A MarginalLogLikelihood class.
        cv_folds: A CVFolds tuple.
        fit_args: Arguments passed along to fit_gpytorch_model

    Returns:
        A CVResults tuple with the following fields

        - model: GPyTorchModel for batched cross validation
        - posterior: GPyTorchPosterior where the mean has shape `n x 1 x m` or
          `batch_shape x n x 1 x m`
        - observed_Y: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of observations.
        - observed_Yvar: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of observed
          measurement noise.

    Example:
        >>> train_X = torch.rand(10, 1)
        >>> train_Y = torch.sin(6 * train_X) + 0.2 * torch.rand_like(train_X)
        >>> cv_folds = gen_loo_cv_folds(train_X, train_Y)
        >>> cv_results = batch_cross_validation(
        >>>     SingleTaskGP,
        >>>     ExactMarginalLogLikelihood,
        >>>     cv_folds,
        >>> )

    WARNING: This function is currently very memory inefficient, use it only
        for problems of small size.
    """
    fit_args = fit_args or {}
    kwargs = {
        "train_X": cv_folds.train_X,
        "train_Y": cv_folds.train_Y,
        "train_Yvar": cv_folds.train_Yvar,
    }
    model_cv = model_cls(**_filter_kwargs(model_cls, **kwargs))
    ic(model_cv)
    model_cv.likelihood.dim=50
    mll_cv = mll_cls(model_cv.likelihood, model_cv)
    mll_cv.to(cv_folds.train_X)
    # mll_cv = fit_gpytorch_model(mll_cv, **fit_args)
    mll_cv = fit_fully_bayesian_model_nuts(
        mll_cv, warmup_steps=256, num_samples=512, thinning=16, disable_progbar=True
    )

    # Evaluate on the hold-out set in batch mode
    with torch.no_grad():
        posterior = model_cv.posterior(
            cv_folds.test_X, observation_noise=observation_noise
        )

    return CVResults(
        model=model_cv,
        posterior=posterior,
        observed_Y=cv_folds.test_Y,
        observed_Yvar=cv_folds.test_Yvar,
    )
import random
def cross_validation_saasbo(x:torch.Tensor,y:torch.Tensor):
    num = list(x.size())[0]
    size_test = math.ceil(num*0.9)
    size_eval = math.floor(num*0.1)
    train_indices = []
    x_train = []
    y_train = []
    i = 0
    while i < size_test:
        j = random.randint(0, num-1)
        if j not in train_indices:
            train_indices.append(j)
            x_train.append(x[j,:])
            y_train.append(y[j])
            i += 1

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train).unsqueeze(1)

    x_test = []
    y_test = []
    for i in range(num):
        if i not in train_indices:
            x_test.append(x[i,:])
            y_test.append(y[i])
    x_test = torch.stack(x_test)
    ic(x_test.shape)
    y_test = torch.tensor(y_test).unsqueeze(1)

    model = SaasFullyBayesianSingleTaskGP(
        train_X=x_train, train_Y=y_train, 
    )
    fit_fully_bayesian_model_nuts(
        model, warmup_steps=256, num_samples=448, thinning=16, disable_progbar=True
    )
 
    with torch.no_grad():
        posterior = model.posterior(
            x_test
        )
    return CVResults(
        model=model,
        posterior=posterior,
        observed_Y=y_test,
        observed_Yvar=None,
    ), y_test

def cross_validation(x,y,yvar=None, is_saasbo=False):
    x = torch.tensor(x,dtype=dtype)

    x = normalize(x,torch.tensor([0,4]))
    y = torch.tensor(y,dtype=dtype)
    y = standardize(y)
    if yvar is None:
        model = SingleTaskGP
        # yvar = torch.zeros(y.size(),dtype=dtype)
    else:
        yvar = torch.tensor(yvar,dtype=dtype)
        model = FixedNoiseGP

    cv_folds = gen_loo_cv_folds(train_X=x, train_Y=y)


    
    if not is_saasbo:
        cv_results = batch_cross_validation(
        model_cls=model,
        mll_cls=ExactMarginalLogLikelihood,
        cv_folds=cv_folds,
        )
        ic(cv_folds.test_Y.squeeze().shape)
        npy = cv_folds.test_Y.squeeze().numpy()

    else:
        # ic(cv_folds)
        ic(cv_folds.train_X.shape)
        ic(cv_folds.train_Y.shape)
        ic(cv_folds.test_X.shape)
        ic(cv_folds.test_Y)
        # cv_results = cross_validation_saasbo(cv_folds=cv_folds)
        # cv_results, test_y = cross_validation_saasbo(x,y)
        # npy = test_y.squeeze().numpy()

    posterior = cv_results.posterior
  

    import json
    with open("model.json", "w") as f:
        json.dump(posterior.__dict__,f)
    mean = posterior.mean
    ic(mean.squeeze().shape)
    
    npmean = mean.squeeze().numpy()
  
    # cv_error = ((cv_folds.test_Y.squeeze() - mean.squeeze()) ** 2).mean()
    Standardize_Outcome
    mape = _mape(npy,npmean)

    # ic(npy)
    # ic(npmean)

    r = _correlation_coefficient(npy, npmean)
    ic(r)
    return r, mape