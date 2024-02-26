from utils.llm_chain import ChainWrapper, get_chain_metadata
from pathlib import Path
from dataset.base_dataset import DatasetBase
import pandas as pd

class LLMEstimator:
    """
    A wrapper for an estimator using LLM
    """

    def __init__(self, opt):
        """
        Initialize a new instance of the LLMEstimator class.
        :param opt: The configuration file (EasyDict)
        """
        self.opt = opt
        self.chain = None
        self.mini_batch_size = opt.mini_batch_size
        self.mode = opt.mode
        self.num_workers = opt.num_workers
        if 'instruction' in opt.keys():
            self.cur_instruct = opt.instruction
        else:
            self.cur_instruct = None

    @staticmethod
    def generate_sample_text(sample_id: int, text: str) -> str:
        """
        Generate a sample text for the chain prompt
        :param sample_id: The sample id
        :param text: The text of the sample
        :return: The sample text for the prompt
        """
        return f"ID: {sample_id};  Sample: {text}\n"

    def calc_usage(self) -> float:
        """"
        Calculate the usage of the estimator
        """
        return self.chain.accumulate_usage

    def init_chain(self, label_schema: set[str]):
        """
        Initialize the chain
        :param label_schema: The label schema
        """
        chain_metadata = get_chain_metadata(Path(self.opt.prompt), retrieve_module=True)
        if hasattr(chain_metadata['module'], 'update_classification_prediction_schema'):
            chain_metadata['json_schema'] = chain_metadata['module'].update_classification_prediction_schema(
                chain_metadata['json_schema'],
                label_schema
            )
        self.chain = ChainWrapper(self.opt.llm, self.opt.prompt, chain_metadata['json_schema'],
                                  chain_metadata['parser_func'])

    def apply_dataframe(self, record: pd.DataFrame):
        """
        Apply the estimator on a dataframe
        :param record: The record
        """
        chain_input = ''
        mini_batch_inputs = []
        record[self.mode] = 'Discarded'
        # prepare all the inputs for the chains
        for i, row in record.iterrows():
            chain_input += self.generate_sample_text(i, row['text'])
            if ((i + 1) % self.mini_batch_size) == 0:
                mini_batch_inputs.append({'batch_size': self.mini_batch_size, 'task_instruction': self.cur_instruct,
                                          'samples': chain_input})
                chain_input = ''
        if not (chain_input == ''):
            mini_batch_inputs.append({'batch_size': self.mini_batch_size, 'task_instruction': self.cur_instruct,
                                      'samples': chain_input})

        all_results = self.chain.batch_invoke(mini_batch_inputs, self.num_workers)
        union_results = [element for sublist in all_results for element in sublist['results']]
        for res in union_results:
            record.loc[res['id'], self.mode] = res['prediction']
        return record

    def apply(self, dataset: DatasetBase, idx: int, leq: bool = False):
        """
        Apply the estimator on the batches up to idx (includes), it then updates the annotation field
        if self.mode is 'annotation', otherwise it update the prediction field.
        :param dataset: The dataset
        :param idx: The current batch index
        :param leq: If True, apply on all the batches up to idx (includes), otherwise apply only on idx
        """
        if self.chain is None:
            self.init_chain(dataset.label_schema)
        if leq:
            batch_records = dataset.get_leq(idx)
        else:
            batch_records = dataset[idx]
        return self.apply_dataframe(batch_records)
