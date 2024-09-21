from utils.llm_chain import ChainWrapper, set_callbck
from dataset.base_dataset import DatasetBase
import pandas as pd
from agent.agent_utils import build_agent, batch_invoke, load_tools
from utils.config import get_llm


class AgentEstimator:
    """
    A wrapper for an estimator of agent
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
        self.usage_callback = set_callbck(opt.llm.type)
        self.agent = None
        if 'instruction' in opt.keys():
            self.cur_instruct = opt.instruction
        else:
            self.cur_instruct = None
        if 'tools' in opt.keys():
            self.tools = opt.tools
        else:
            self.tools = load_tools(opt.tools_path)
        self.llm = get_llm(opt.llm)
        self.chain_yaml_extraction = ChainWrapper(opt.llm, 'prompts/meta_prompts_agent/extract_yaml.prompt', None, None)
        self.total_usage = 0

    def calc_usage(self) -> float:
        """"
        Calculate the usage of the estimator
        """
        return self.total_usage

    def apply_dataframe(self, record: pd.DataFrame):
        """
        Apply the estimator on a dataframe
        :param record: The record
        """
        batch_inputs = []
        # prepare all the inputs for the chains
        for i, row in record.iterrows():
            batch_inputs.append({'input': row['text']})
        all_results = batch_invoke(self.agent.invoke, batch_inputs, self.num_workers, self.usage_callback)
        self.total_usage += sum([res['usage'] for res in all_results])
        for res in all_results:
            record.loc[res['index'], self.mode] = res['error'] if res['result'] is \
                                                                   None else self.sample_output_to_str(res['result'])
        return record

    @staticmethod
    def sample_output_to_str(sample_output: dict) -> str:
        """
        Convert the sample output to a string
        :param sample_output: The sample output
        :return: The string representation
        """
        if 'text' in sample_output.keys():
            return sample_output['text']
        if 'intermediate_steps' not in sample_output.keys():
            return sample_output['text']
        intermediate_str = ''
        for i, intermediate in enumerate(sample_output['intermediate_steps']):
            intermediate_str += f"#Intermediate step {i + 1}: {intermediate[0].log[:-2]}"
            if isinstance(intermediate[1], str) or isinstance(intermediate[1], int) or isinstance(intermediate[1], float):
                intermediate_str += f"#Result step {i + 1}: {intermediate[1]}\n"
            elif 'result' in intermediate[1].keys():
                intermediate_str += f"#Result step {i + 1}: {intermediate[1]['result']}\n"
        output = sample_output['output']
        return f"##Agent intermediate steps:\n{intermediate_str}\n##Agent final output:\n```yaml {output}```"

    def apply(self, dataset: DatasetBase, idx: int, leq: bool = False):
        """
        Apply the estimator on the batches up to idx (includes), it then updates the annotation field
        if self.mode is 'annotation', otherwise it update the prediction field.
        :param dataset: The dataset
        :param idx: The current batch index
        :param leq: If True, apply on all the batches up to idx (includes), otherwise apply only on idx
        """

        self.agent = build_agent(self.llm, self.tools,
                                 self.cur_instruct, intermediate_steps=True)

        if leq:
            batch_records = dataset.get_leq(idx)
        else:
            batch_records = dataset[idx]
        return self.apply_dataframe(batch_records)
