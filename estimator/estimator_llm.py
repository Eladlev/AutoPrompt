from utils.llm_chain import ChainWrapper
import json
import asyncio
from tqdm import trange
from dataset.base_dataset import DatasetBase


class LLMEstimator:
    """
    A wrapper for an estimator using LLM
    """

    def __init__(self, opt):
        """
        Initialize a new instance of the LLMEstimator class.
        :param opt: The configuration file (EasyDict)
        """
        self.chain = ChainWrapper(opt.llm, opt.prompt, json.loads(opt.json_schema))
        self.mini_batch_size = opt.mini_batch_size
        self.mode = opt.mode
        self.num_workers = opt.num_workers
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

    def apply(self, dataset: DatasetBase, idx: int, leq: bool = True):
        """
        Apply the estimator on the batches up to idx (includes), it then updates the annotation field
        if self.mode is 'annotation', otherwise it update the prediction field.
        :param dataset: The dataset
        :param idx: The current batch index
        :param leq: If True, apply on all the batches up to idx (includes), otherwise apply only on idx
        """
        if leq:
            batch_records = dataset.get_leq(idx)
        else:
            batch_records = dataset[idx]
        chain_input = ''
        mini_batch_inputs = []

        # prepare all the inputs for the chains
        for i, row in batch_records.iterrows():
            chain_input += self.generate_sample_text(i, row['text'])
            if ((i + 1) % self.mini_batch_size) == 0:
                mini_batch_inputs.append({'batch_size': self.mini_batch_size, 'task_instruction': self.cur_instruct,
                                          'samples': chain_input})
                chain_input = ''
        if not (chain_input == ''):
            mini_batch_inputs.append({'batch_size': self.mini_batch_size, 'task_instruction': self.cur_instruct,
                                      'samples': chain_input})

        # run the chains (either in parallel or serially)
        for i in trange(0, len(mini_batch_inputs), self.num_workers, desc='Predicting'):
            if self.num_workers > 1:
                results = asyncio.run(self.chain.async_batch_invoke(mini_batch_inputs[i:i + self.num_workers]))
                all_results = []
                for res in results:
                    all_results += res['results']
            else:
                results = self.chain.invoke(mini_batch_inputs[i])
                all_results = results['results']
            for res in all_results:
                batch_records.loc[res['id'], self.mode] = res['prediction']
        dataset.update(batch_records)
