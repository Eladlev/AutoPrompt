from utils.llm_chain import ChainWrapper
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
import copy

from utils.dedup import Dedup

class MetricMetadata(BaseModel):
    metric_score: float = Field(description="The score for the metric")
    metric_reason: str = Field(description="The reason for the metric score")


class MetricHandler:
    """
    A Class responsible for handling and generating metrics
    """

    def __init__(self, config, chains: dict, task_metadata: dict):
        """
        Initialize a new instance of the MetricHandler class.
        :param config: The configuration file (EasyDict)
        :metric_generator: The metric generator chain
        :task_metadata: The task metadata
        """
        self.config = config
        self.metric_generator = chains['metric_generator']
        self.metric_merge = chains['metric_merge']

        self.dedup = Dedup(self.config)
        self.task_metadata = task_metadata
        self.metrics = config.get('metrics', [])
        self.num_metrics = self.config.num_metrics
        init_metrics = config.get('init_metrics', True)
        if init_metrics:
            self.generate_metrics()

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

    def chunk_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

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


    def get_semantic_metric_clusters(self, metrics) -> list[list[int]]:
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
        result = []
        for cluster in metric_clusters:
            if len(cluster) > 1:
                cur_cluster_metrics = copy.deepcopy([metrics[i] for i in cluster])
                chain_params = copy.deepcopy(self.task_metadata)
                for metric in cur_cluster_metrics:
                    metric.pop('metric_prompt')
                cur_cluster_metrics = {metric['metric_name']: metric['metric_desc'] for metric in cur_cluster_metrics}

                chain_params.update({'num_metrics': self.num_metrics,
                                     'metrics_to_merge': self.metric_to_text(cur_cluster_metrics)})
                merged_metric = self.metric_merge.invoke(chain_params)
                merged_metric = merged_metric['metrics_list']
                self.update_metrics(merged_metric)
                result += merged_metric
            else:
                result.append(metrics[cluster[0]])
        return result

    def get_metrics(self) -> List[dict]:
        """
        Get the metrics
        """

        return [{k: v for k, v in d.items() if k != 'metric_function'} for d in self.metrics]

    def generate_metrics(self) -> dict:
        """
        Generate new metrics
        """
        number_of_remaining_metrics = self.num_metrics - len(self.metrics)
        if number_of_remaining_metrics > 0:
            chain_params = copy.deepcopy(self.task_metadata)
            chain_params.update({'num_metrics': number_of_remaining_metrics})
            #TODO: provide the predefined metrics as input for the generation (remove redundant metrics)
            metrics = self.metric_generator.invoke(chain_params)
            metrics = metrics['metrics_list']
            self.update_metrics(metrics)
            metric_clusters = self.get_semantic_metric_clusters(metrics)
            metric_clusters = [list(cluster) for cluster in metric_clusters]
            # For each cluster, we prune and reduce the number of metrics
            self.metrics += self.sample_metrics(metrics, metric_clusters)

        for metric in self.metrics:
            if 'task_tools_description' in self.task_metadata.keys():
                prompt = f"""{metric["metric_prompt"]}\n\n\nThe following input should be evaluated according to the metric guidelines. The input consists of
1. User request
2. The agent response including the tools usage with all the intermediate steps and the model final output.
###Evaluated input:\n{{sample}}\n###End"""
            else:
                prompt = f'{metric["metric_prompt"]}\nThe following input which consists of user request and model response, should be evaluated according to the metric guidelines. \n###Evaluated input:\n{{sample}}\n###End'
            metric['metric_function'] = self.build_score_function(prompt)

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
