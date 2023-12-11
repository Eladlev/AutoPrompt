from .estimator_argilla import ArgillaEstimator
from .estimator_llm import LLMEstimator
from .estimator_llm_batch import LLMBatchEstimator


def give_estimator(opt):
    if opt.method == 'argilla':
        return ArgillaEstimator(opt.config)
    elif opt.method == 'llm':
        return LLMEstimator(opt.config)
    elif opt.method == 'llm_batch':
        return LLMBatchEstimator(opt.config)
    else:
        raise NotImplementedError
