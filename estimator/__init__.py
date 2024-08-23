import pandas as pd

from .estimator_argilla import ArgillaEstimator
from .estimator_llm import LLMEstimator
from .estimator_t2i import T2IEstimator
from .estimator_vlm import VLMEstimator
from .estimator_llm_batch import LLMBatchEstimator
from .estimator_agent import AgentEstimator
from dataset.base_dataset import DatasetBase


class DummyEstimator:
    """
    A dummy callback for the Estimator class.
    This is a method to handle an empty estimator.
    """

    @staticmethod
    def calc_usage():
        """
        Dummy function to calculate the usage of the dummy estimator
        """
        return 0

    @staticmethod
    def apply(dataset: DatasetBase, batch_id: int):
        """
        Dummy function to mimic the apply method, returns an empty dataframe
        """
        return pd.DataFrame()


def give_estimator(opt):
    if opt.method == 'argilla':
        return ArgillaEstimator(opt.config)
    elif opt.method == 'llm':
        return LLMEstimator(opt.config)
    elif opt.method == 'llm_batch':
        return LLMBatchEstimator(opt.config)
    elif opt.method == 'agent':
        return AgentEstimator(opt.config)
    elif opt.method == 't2i':
        return T2IEstimator(opt.config)
    elif opt.method == 't2i_batch':
        raise NotImplementedError("Batch image mode not yet implemented")
    elif opt.method == 'vlm':
        return DummyEstimator()
        # return VLMEstimator(opt.config)
    else:
        return DummyEstimator()
