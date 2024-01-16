from estimator.estimator_llm import LLMEstimator
from dataset.base_dataset import DatasetBase
import pandas as pd


class LLMBatchEstimator:
    """
    A wrapper for an estimator using aggregation of multiple LLMs estimators
    """

    def __init__(self, opt):
        """
        Initialize a new instance of the LLMEstimator class.
        :param opt: The configuration file (EasyDict)
        """
        self.llm_estimators = [LLMEstimator(opt.estimator_config) for _ in range(len(opt.instructions))]
        for i, estimator in enumerate(self.llm_estimators):
            estimator.cur_instruct = opt.instructions[i]
        self.mode = opt.estimator_config.mode
        self.aggregation_mode = opt.aggregation_mode

    def calc_usage(self) -> float:
        """"
        Calculate the usage of the estimator
        """
        return sum([estimator.calc_usage() for estimator in self.llm_estimators])

    def get_aggregation_function(self):
        if self.aggregation_mode == 'max':
            return lambda record: max(record)
        elif self.aggregation_mode == 'min':
            return lambda record: min(record)
        elif self.aggregation_mode == 'mean':
            return lambda record: sum(record) / len(record)
        elif self.aggregation_mode == 'median':
            return lambda record: sorted(record)[len(record) // 2]
        elif self.aggregation_mode == 'majority':
            return lambda record: max(set(record), key=record.count)
        elif self.aggregation_mode == 'exist':
            return lambda record: 'Yes' if any([t == 'Yes' for t in record]) else 'No'
        elif self.aggregation_mode == 'all':
            return lambda record: 'Yes' if all([t == 'Yes' for t in record]) else 'No'
        else:
            raise Exception(f'Unknown aggregation class {self.aggregation_mode}')

    def apply(self, dataset: DatasetBase, idx: int, leq: bool = False):
        """
        Apply the estimator on the batches up to idx (includes), it then updates the annotation field
        if self.mode is 'annotation', otherwise it update the prediction field.
        :param dataset: The dataset
        :param idx: The current batch index
        :param leq: If True, apply on all the batches up to idx (includes), otherwise apply only on idx
        """
        update_datasets = [estimator.apply(dataset, idx, leq) for estimator in self.llm_estimators]
        res_dataset = update_datasets[0]
        if res_dataset.empty:
            return res_dataset
        for i, df in enumerate(update_datasets[1:]):
            # Merge the dataframes on the 'id' column
            merged_df = pd.merge(res_dataset, df[['id', self.mode]], on='id', how='left', suffixes=('_left', '_right'))
            if i == 0:
                res_dataset[self.mode] = merged_df.apply(lambda row: [str(row['{}_left'.format(self.mode)])] +
                                                                     [str(row['{}_right'.format(self.mode)])], axis=1)
            else:
                res_dataset[self.mode] = merged_df.apply(lambda row: row['{}_left'.format(self.mode)] +
                                                                     [str(row['{}_right'.format(self.mode)])], axis=1)
        res_dataset[self.mode] = res_dataset[self.mode].apply(self.get_aggregation_function())
        return res_dataset
