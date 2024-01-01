from utils.llm_chain import ChainWrapper, get_chain_metadata
from pathlib import Path
import asyncio
from tqdm import trange, tqdm
from dataset.base_dataset import DatasetBase
import concurrent.futures


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

    def apply(self, dataset: DatasetBase, idx: int, leq: bool = True):
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
        def sample_generator():
            for sample in mini_batch_inputs:
                yield sample

        def process_sample_with_progress(sample):
            result = self.chain.invoke(sample)
            pbar.update(1)  # Update the progress bar
            return result

        if not('async_params' in self.opt.llm.keys()): #non async mode, use regular workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                with tqdm(total=len(mini_batch_inputs), desc="Processing samples") as pbar:
                    all_results = list(executor.map(process_sample_with_progress, sample_generator()))
                for res in all_results:
                    for sample_res in res['results']:
                        batch_records.loc[sample_res['id'], self.mode] = sample_res['prediction']
        else:
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
        return batch_records
