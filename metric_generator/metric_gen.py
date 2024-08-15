from utils.llm_chain import ChainWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
from utils.dedup import Dedup

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
        self.dedup = Dedup(self.config)
        self.metrics = self.generate_metrics()

    def get_metrics_info(self) -> dict:
        """
        Get the metrics
        """
        return {t['metric_name']: t['metric_desc'] for t in self.metrics}

    def update_metrics(self, metrics_list) -> dict:
        """
        Update metrics dictionary to merge metric_scale into metric_prompt
        """
        for metric_dict in metrics_list:
            for _, eval_prompt in metric_dict['metric_scale'][0].items():
                metric_dict['metric_prompt'] += f'\n{eval_prompt}'
            del metric_dict['metric_scale']
            
    def hard_filter_metrics(self, metrics) -> list[bool]:
        """
        Produces a list of boolean values corresponding to whether the metric should be dropped (0) or not (1)
        based on relevance to task description
        Args:
            metrics (list[dict]): list of metric dictionaries containing metric_name, metric_desc and metric_prompt
        Returns:
            list[bool]: a list of boolean values indicating whether the metric at that index should be kept (1) or dropped (0), referencing to the metrics list above
        """
        pass
    
    def get_semantic_metric_clusters(self, metrics) -> list[set(int)]:
        """
        Groups metrics in clusters based on semantic similarity using the Dedup class
        Args:
            metrics (list[dict]): list of metric dictionaries containing metric_name, metric_desc and metric_prompt
        """
        new_dedup = self.dedup.copy()
        metrics_text = []
        for metric_subdict in metrics:
            curr_metric_text = f"{metric_subdict['metric_name']}. {metric_subdict['metric_desc']}"
            metrics_text.append(curr_metric_text)
        records = pd.DataFrame(metrics_text, columns=['text'])
        return new_dedup.cluster_data(records)
    
    def sample_metrics(self, metrics, metric_clusters) -> list[dict]:
        """
        Samples a single metric for each metric cluster with size > 1. For clusters of size = 1, the single metric is kept as is.
        The sample metric for a size > 1 cluster is synthetically generated using an LLM to be most representative of the metric clusters.

        Args:
            metrics (list[dict]): list of metric dictionaries containing metric_name, metric_desc and metric_prompt
            metric_clusters (list[list[int]]): list of metric cluster, where each sublist is an individual cluster containing the 0-indexed indices of the metrics that belong to a given cluster, referencing to the original metrics list
        Returns:
            list[dict]: the new list of metric dictionaries containing metric_name, metric_desc and metric_prompt
        """
        pass
    

    def generate_metrics(self) -> dict:
        """
        Generate new metrics
        """
        metrics = self.metric_generator.invoke(
            {'task_description': self.task_description, 'num_metrics': self.config.num_metrics})
        metrics = metrics['metrics_list']
        
        self.update_metrics(metrics)
        
        metrics_filter = self.hard_filter_metrics(metrics)
        
        metrics[:] = [metric_subdict for metric_subdict, keep_metric in zip(metrics, metrics_filter) if keep_metric]
        
        metric_clusters = self.get_semantic_metric_clusters(metrics)
        metric_clusters = [list(cluster) for cluster in metric_clusters]
        
        metrics = self.sample_metrics(metrics, metric_clusters)
        
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
