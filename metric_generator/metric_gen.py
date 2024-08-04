from utils.llm_chain import ChainWrapper
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
        metrics = self.metric_generator.invoke({'task_description': self.task_description})
        return metrics.dict()['metrics_list']
