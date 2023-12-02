from .estimator_argilla import ArgillaEstimator
from .estimator_llm import LLMEstimator


def give_estimator(opt):
    if opt.method == 'argilla':
        return ArgillaEstimator(opt.config)
    elif opt.method == 'llm':
        return LLMEstimator(opt.config)
    else:
        raise NotImplementedError
