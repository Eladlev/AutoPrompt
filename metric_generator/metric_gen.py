from utils.llm_chain import ChainWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
import copy


class MetricMetadata(BaseModel):
    metric_score: float = Field(description="The score for the metric")
    metric_reason: str = Field(description="The reason for the metric score")


class MetricHandler:
    """
    A Class responsible for handling and generating metrics
    """

    def __init__(self, config, metric_generator: ChainWrapper, task_metadata: dict):
        """
        Initialize a new instance of the MetricHandler class.
        :param config: The configuration file (EasyDict)
        :metric_generator: The metric generator chain
        :task_metadata: The task metadata
        """
        self.config = config
        self.metric_generator = metric_generator
        self.task_metadata = task_metadata
        self.metrics = self.generate_metrics()

    def get_metrics_info(self, as_text=True) -> dict or str:
        """
        Get the metrics
        """
        metric_dic = {t['metric_name']: t['metric_desc'] for t in self.metrics}
        if as_text:
            return self.metric_to_text(metric_dic)
        return metric_dic

    @staticmethod
    def metric_to_text(metric: dict) -> str:
        """
        Convert a metric to text
        :param metric: The metric dictionary
        :return: All metric info as text
        """
        text = '####Metrics info:\n'
        for key, value in metric.items():
            text += f'{key}: {value}\n##\n'
        text += '####End of metrics info\n'
        return text

    def update_metrics(self, metrics_list):
        """
        Update metrics dictionary to merge metric_scale into metric_prompt
        """
        # TODO: This is hardcoded, should be extracted from the schema
        score_translation = {'deficient_desc': 1, 'adequate_desc': 2, 'competent_desc': 3,
                             'proficient_desc': 4, 'exemplary_desc': 5}
        for metric_dict in metrics_list:
            for metric_key, eval_prompt in metric_dict['metric_scale'][0].items():
                metric_dict['metric_prompt'] += f'\nscore {score_translation[metric_key]}: {eval_prompt}'
            del metric_dict['metric_scale']

    def generate_metrics(self) -> dict:
        """
        Generate new metrics
        """
        chain_params = copy.deepcopy(self.task_metadata)
        chain_params.update({'num_metrics': self.config.num_metrics})
        metrics = self.metric_generator.invoke(chain_params)
        metrics = metrics['metrics_list']
        self.update_metrics(metrics)
        for metric in metrics:
            prompt = f'{metric["metric_prompt"]}\nThe following input should be evaluated according to the metric guidelines. \n###Evaluated input:\n{{sample}}\n###End'
            metric['metric_function'] = self.build_score_function(prompt)
        return metrics

    def build_score_function(self, metric_prompt: str):
        """
        Constructs a scoring function based on the provided configuration.

        This function initializes a ChainWrapper with the given configuration, prompt file, and MetricMetadata.
        It then defines a new function that invokes the chain with the input prompt and returns the results.

        :param metric_prompt: The prompts.
        :return: A function that takes an input prompt, invokes the chain, and returns the results.
        """

        chain = ChainWrapper(self.config.llm, metric_prompt, MetricMetadata)

        def new_function(record: pd.DataFrame, num_workers: int = 1):
            batch_inputs = []
            # prepare all the inputs for the chains
            for i, row in record.iterrows():
                batch_inputs.append({'sample': row['text']})
            all_results = chain.batch_invoke(batch_inputs, num_workers, get_index=True)
            all_results = {res['index']: res['result'].dict() for res in all_results}
            return all_results

        return new_function
