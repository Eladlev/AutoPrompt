from utils.llm_chain import ChainWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
import pickle
class MetricMetadata(BaseModel):
    metric_score: float = Field(description="The score for the metric")
    metric_reason: str = Field(description="The reason for the metric score")


class MetricHandler:
    """
    A Class responsible for handling and generating metrics
    """

    def __init__(self, config, metric_generator: ChainWrapper, task_description: str):
        """
        Initialize a new instance of the MetricHandler class.
        :param config: The configuration file (EasyDict)
        :metric_generator: The metric generator chain
        :task_description: Describe the task that needed to be solved
        """
        self.config = config
        self.metric_generator = metric_generator
        self.task_description = task_description
        self.metrics = self.generate_metrics()

    def generate_metrics(self) -> dict:
        """
        Generate new metrics
        """
        metrics = self.metric_generator.invoke(
            {'task_description': self.task_description, 'num_metrics': self.config.num_metrics})
        metrics = metrics['metrics_list']
        for metric in metrics:
            prompt = f'{metric["metric_prompt"]}\nThe following input should be evaluated according to the metric guidlines. \n###Evaluated input:\n{{sample}}\n###End'
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
