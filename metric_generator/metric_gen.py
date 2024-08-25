from utils.llm_chain import ChainWrapper
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
from utils.dedup import Dedup

class MetricMetadata(BaseModel):
    metric_score: float = Field(description="The score for the metric")
    metric_reason: str = Field(description="The reason for the metric score")

class MetricEvaluation(BaseModel):
    include_metrics: list[int] = Field(...)

class MetricScale(BaseModel):
    deficient_desc: str
    adequate_desc: str
    competent_desc: str
    proficient_desc: str
    exemplary_desc: str

class Metric(BaseModel):
    metric_name: str
    metric_desc: str
    metric_prompt: str
    metric_scale: List[MetricScale]

class MetricScores(BaseModel):
    metrics_list: List[Metric]

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
    
    def merge_metrics(self, metrics, cluster, task_description):
        metrics_to_merge = [metrics[i] for i in cluster]

        prompt = f"""You are an AI expert tasked with merging multiple related metrics into a single, comprehensive metric. Your goal is to create a new metric that captures the essence of all the input metrics while eliminating redundancy.

                Given the following set of metrics:

                {metrics_to_merge}

                Create a single new metric that:
                1. Combines the key aspects of all input metrics
                2. Has a clear and concise name
                3. Provides a comprehensive description
                4. Includes a well-formulated evaluation prompt

                Ensure that the new metric is relevant to task description :
                {task_description}

                Please return a single metric.
                """
        chain = ChainWrapper(self.config.llm, prompt, MetricScores)
        response = chain.invoke({'metrics_to_merge': metrics_to_merge,'task_description': task_description})
        return response.metrics_list
            
    def hard_filter_metrics(self, metrics) -> list[bool]:
        """
        Produces a list of boolean values corresponding to whether the metric should be dropped (0) or not (1)
        based on relevance to task description
        Args:
            metrics (list[dict]): list of metric dictionaries containing metric_name, metric_desc and metric_prompt
        Returns:
            list[bool]: a list of boolean values indicating whether the metric at that index should be kept (1) or dropped (0), referencing to the metrics list above
        """
        chunked_metrics = [metrics[i:i + 5] for i in range(0, len(metrics), 5)]
        task_description = self.task_description
        all_evaluations = []
        batch_inputs = []
        metric_selection_prompt = f'''You are tasked with evaluating the consistency of metrics based on their descriptions and prompts. Your task is to analyze a set of metrics for their relevance, uniqueness, and measurability in the context of evaluating an AI assistant's performance.
                Evaluate each metric based on the provided task descriptions, metric description and associated prompts with metrics. If the descriptions of prompts is logically consistent and clear, include the metric. If the combination is inconsistent or unclear, exclude the metric from the output.
                For each metric, consider the following criteria:
                1. Relevance: Is the metric directly related to assessing the performance of an AI assistant?
                2. Uniqueness: Does the metric measure an aspect that is not already covered by other metrics in the set?
                3. Measurability: Can the metric be objectively measured based on the provided scale and prompt?
                4. Clarity: Is the metric description and prompt clear and unambiguous?
                Please analyze each metric and task description and return a list of exactly {len(chunk)} integers where 1 indicates the metric should be included and 0 indicates it should be removed.
                ###Task Descriptions: \n
                {task_description}\n
                ###Metric Descriptions and Prompts: \n
                {chunk}\n'''
        for chunk in chunked_metrics:
            batch_inputs.append({'chunk': chunk, 'task_description': self.task_description})
        chain = ChainWrapper(self.config.llm, metric_selection_prompt, MetricEvaluation)
        all_results = chain.batch_invoke(batch_inputs, num_workers=1, get_index=True)
        all_results = [res['result'].include_metrics for res in all_results]
        for result in all_results:
            all_evaluations.extend(result)
        return all_evaluations

    
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
        result = []
        for cluster in metric_clusters:
            if len(cluster) > 1:
                merged_metric = self.merge_metrics(metrics, cluster, self.task_description)
                result.append(merged_metric)
            else:
                result.append(metrics[cluster[0]])
        return result
    

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
