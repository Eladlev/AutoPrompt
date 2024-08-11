from dataset.base_dataset import DatasetBase
from easydict import EasyDict as edict


class SampleGenerator:
    """
    This class is responsible for generating samples for the dataset.
    """

    def __init__(self, config: edict, task_description: str, meta_chain: dict):
        """
        Initialize a new instance of the SampleGenerator class.
        :param config: The configuration file (EasyDict)
        :param task_description: The task description
        :param meta_chain: The meta chain
        """
        self.config = config
        self.task_description = task_description
        self.meta_chain = meta_chain

    def metric_to_text(self, metric: dict) -> str:
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
    def generate_initial_samples(self, dataset: DatasetBase, cur_prompt: str, metrics_info: dict = None):
        """
        In case the initial dataset is empty generate the initial samples
        :param dataset: The dataset
        :param cur_prompt: The current prompt
        :param metrics_info: The metrics information
        """

        batch_input = {"num_samples": self.config.samples_generation_batch,
                       "task_description": self.task_description,
                       "instruction": cur_prompt}
        if metrics_info is not None:
            metrics_text = self.metric_to_text(metrics_info)
            batch_input['metrics_info'] = metrics_text
        batch_inputs = self.generate_samples_batch(batch_input, self.config.num_initialize_samples,
                                                   self.config.samples_generation_batch)

        samples_batches = self.meta_chain.chain.initial.batch_invoke(batch_inputs, self.config.num_workers)
        samples_list = [element for sublist in samples_batches for element in sublist['samples']]
        samples_list = dataset.remove_duplicates(samples_list)
        dataset.add(samples_list, 0)

    def generate_samples(self, dataset: DatasetBase, cur_prompt: str, history: list, batch_id: int,
                         samples_to_text, metrics_info: dict = None):
        """
        Generate samples for the dataset
        :param dataset: The dataset
        :param cur_prompt: The current prompt
        :param history: The history of the samples
        :param batch_id: The batch id
        :param samples_to_text: The function to convert samples to text
        :param metrics_info: The metrics information
        """
        batch_input = {"num_samples": self.config.samples_generation_batch,
                       "task_description": self.task_description,
                       "prompt": cur_prompt}
        batch_inputs = self.generate_samples_batch(batch_input, self.config.num_generated_samples,
                                                   self.config.samples_generation_batch)

        if sum([len(t['errors']) for t in history]) > 0:
            history_samples = '\n'.join([samples_to_text(sample,
                                                         num_errors_per_label=self.config.num_err_samples,
                                                         is_score=False) for sample in history])
            for batch in batch_inputs:
                extra_samples = dataset.sample_records()
                extra_samples_text = DatasetBase.samples_to_text(extra_samples)
                batch['history'] = history_samples
                batch['extra_samples'] = extra_samples_text
        else:
            for batch in batch_inputs:
                extra_samples = dataset.sample_records()
                extra_samples_text = DatasetBase.samples_to_text(extra_samples)
                batch['history'] = 'No previous errors information'
                batch['extra_samples'] = extra_samples_text

        samples_batches = self.meta_chain.chain.step_samples.batch_invoke(batch_inputs,
                                                                          self.config.num_workers)
        new_samples = [element for sublist in samples_batches for element in sublist['samples']]
        new_samples = dataset.remove_duplicates(new_samples)
        dataset.add(new_samples, batch_id)

    @staticmethod
    def generate_samples_batch(batch_input, num_samples, batch_size):
        """
        Generate samples in batch
        """
        batch_num = num_samples // batch_size
        all_batches = [batch_input.copy() for _ in range(batch_num)]
        reminder = num_samples - batch_num * batch_size
        if reminder > 0:
            all_batches.append(batch_input.copy())
            all_batches[-1]['num_samples'] = reminder
        return all_batches