import pandas as pd

from eval.evaluator import Eval
from dataset.base_dataset import DatasetBase
from utils.llm_chain import MetaChain
from estimator import give_estimator
from pathlib import Path
import pickle
import os
import json
import logging
import wandb


class OptimizationPipeline:
    """
    The main pipeline for optimization. The pipeline is composed of 4 main components:
    1. dataset - The dataset handle the data including the annotation and the prediction
    2. annotator - The annotator is responsible generate the GT
    3. predictor - The predictor is responsible to generate the prediction
    4. eval - The eval is responsible to calculate the score and the large errors
    """

    def __init__(self, config, task_description: str = None, initial_prompt: str = None, output_path: str = ''):
        """
        Initialize a new instance of the ClassName class.
        :param config: The configuration file (EasyDict)
        :param task_description: Describe the task that needed to be solved
        :param initial_prompt: Provide an initial prompt to solve the task
        :param output_path: The output dir to save dump, by default the dumps are not saved
        """

        if config.use_wandb:  # In case of using W&B
            wandb.login()
            self.wandb_run = wandb.init(
                project="AutoGPT",
                config=config,
            )
        if output_path == '':
            self.output_path = None
        else:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            self.output_path = Path(output_path)
            logging.basicConfig(filename=self.output_path / 'info.log', level=logging.DEBUG,
                                format='%(asctime)s - %(levelname)s - %(message)s', force=True)

        self.dataset = None
        self.config = config
        self.meta_chain = MetaChain(config)
        self.initialize_dataset()

        self.task_description = task_description
        self.cur_prompt = initial_prompt

        self.predictor = give_estimator(config.predictor)
        self.annotator = give_estimator(config.annotator)
        self.eval = Eval(config.eval, self.meta_chain.error_analysis, self.dataset.label_schema)
        self.batch_id = 0
        self.patient = 0

    @staticmethod
    def log_and_print(message):
        print(message)
        logging.info(message)

    def initialize_dataset(self):
        """
        Initialize the dataset: Either empty dataset or loading an existing dataset
        """
        logging.info('Initialize dataset')
        self.dataset = DatasetBase(self.config.dataset)
        if 'initial_dataset' in self.config.dataset.keys():
            logging.info(f'Load initial dataset from {self.config.dataset.initial_dataset}')
            self.dataset.load_dataset(self.config.dataset.initial_dataset)

    def calc_usage(self):
        """
        Calculate the usage of the optimization process (either $ in case of openAI or #tokens the other cases)
        """
        total_usage = 0
        total_usage += self.meta_chain.calc_usage()
        total_usage += self.annotator.calc_usage()
        total_usage += self.predictor.calc_usage()
        return total_usage

    def extract_best_prompt(self):
        sorted_history = sorted(
            self.eval.history[min(self.config.meta_prompts.warmup - 1, len(self.eval.history) - 1):],
            key=lambda x: x['score'],
            reverse=False)
        return {'prompt': sorted_history[-1]['prompt'], 'score': sorted_history[-1]['score']}

    def run_step_prompt(self):
        """
        Run the meta-prompts and get new prompt suggestion, estimated prompt score and a set of challenging samples
        for the new prompts
        """
        step_num = len(self.eval.history)
        if (step_num < self.config.meta_prompts.warmup) or (step_num % 3) > 0:
            last_history = self.eval.history[-self.config.meta_prompts.history_length:]
        else:
            sorted_history = sorted(self.eval.history[self.config.meta_prompts.warmup - 1:], key=lambda x: x['score'],
                                    reverse=False)
            last_history = sorted_history[-self.config.meta_prompts.history_length:]
        history_prompt = '\n'.join([self.eval.sample_to_text(sample,
                                                        num_errors_per_label=self.config.meta_prompts.num_err_prompt,
                                                        is_score=True) for sample in last_history])
        prompt_input = {"history": history_prompt, "task_description": self.task_description,
                        'error_analysis': last_history[-1]['analysis']}
        if 'label_schema' in self.config.dataset.keys():
            prompt_input["labels"] = json.dumps(self.config.dataset.label_schema)
        prompt_suggestion = self.meta_chain.step_prompt_chain.invoke(prompt_input)
        self.log_and_print(f'Previous prompt score:\n{self.eval.mean_score}\n#########\n')
        self.log_and_print(f'Get new prompt:\n{prompt_suggestion["prompt"]}')
        self.batch_id += 1
        if len(self.dataset) < self.config.dataset.max_samples:
            batch_input = {"num_samples": self.config.meta_prompts.samples_generation_batch,
                           "task_description": self.task_description,
                           "prompt": prompt_suggestion['prompt']}
            batch_inputs = self.generate_samples_batch(batch_input, self.config.meta_prompts.num_generated_samples,
                                                       self.config.meta_prompts.samples_generation_batch)

            if sum([len(t['errors']) for t in last_history]) > 0:
                history_samples = '\n'.join([self.eval.sample_to_text(sample,
                                                                 num_errors_per_label=self.config.meta_prompts.num_err_samples,
                                                                 is_score=False) for sample in last_history])
                for batch in batch_inputs:
                    extra_samples = self.dataset.sample_records()
                    extra_samples_text = DatasetBase.samples_to_text(extra_samples)
                    batch['history'] = history_samples
                    batch['extra_samples'] = extra_samples_text
            else:
                for batch in batch_inputs:
                    extra_samples = self.dataset.sample_records()
                    extra_samples_text = DatasetBase.samples_to_text(extra_samples)
                    batch['history'] = 'No previous errors information'
                    batch['extra_samples'] = extra_samples_text

            samples_batches = self.meta_chain.step_samples.batch_invoke(batch_inputs,
                                                                         self.config.meta_prompts.num_workers)
            new_samples = [element for sublist in samples_batches for element in sublist['samples']]
            new_samples = self.dataset.remove_duplicates(new_samples)
            self.dataset.add(new_samples, self.batch_id)
            logging.info('Get new samples')
        self.cur_prompt = prompt_suggestion['prompt']

    def stop_criteria(self):
        """
        Check if the stop criteria holds. The conditions for stopping:
        1. Usage is above the threshold
        2. There was no improvement in the last > patient steps
        """
        if 0 < self.config.stop_criteria.max_usage < self.calc_usage():
            return True
        if len(self.eval.history) <= self.config.meta_prompts.warmup:
            self.patient = 0
            return False
        min_batch_id, max_score = self.eval.get_max_score(self.config.meta_prompts.warmup-1)
        if max_score - self.eval.history[-1]['score'] > -self.config.stop_criteria.min_delta:
            self.patient += 1
        else:
            self.patient = 0
        if self.patient > self.config.stop_criteria.patience:
            return True
        return False

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

    def generate_initial_samples(self):
        """
        In case the initial dataset is empty generate the initial samples
        """
        batch_input = {"num_samples": self.config.meta_prompts.samples_generation_batch,
                       "task_description": self.task_description,
                       "instruction": self.cur_prompt}
        batch_inputs = self.generate_samples_batch(batch_input, self.config.meta_prompts.num_initialize_samples,
                                                   self.config.meta_prompts.samples_generation_batch)

        samples_batches = self.meta_chain.initial_chain.batch_invoke(batch_inputs, self.config.meta_prompts.num_workers)
        samples_list = [element for sublist in samples_batches for element in sublist['samples']]
        samples_list = self.dataset.remove_duplicates(samples_list)
        self.dataset.add(samples_list, 0)

    def save_state(self):
        """
        Save the process state
        """
        if self.output_path is None:
            return
        logging.info('Save state')
        self.dataset.save_dataset(self.output_path / 'dataset.csv')
        state = {'history': self.eval.history, 'batch_id': self.batch_id,
                 'prompt': self.cur_prompt, 'task_description': self.task_description,
                 'patient': self.patient}
        pickle.dump(state, open(self.output_path / 'history.pkl', 'wb'))

    def load_state(self, path: str):
        """
        Load pretrain state
        """
        path = Path(path)
        if (path / 'dataset.csv').is_file():
            self.dataset.load_dataset(path / 'dataset.csv')
        if (path / 'history.pkl').is_file():
            state = pickle.load(open(path / 'history.pkl', 'rb'))
            self.eval.history = state['history']
            self.batch_id = state['batch_id']
            self.cur_prompt = state['prompt']
            self.task_description = state['task_description']
            self.patient = state['patient']

    def step(self, current_iter, total_iter):
        """
        This is the main optimization process step.
        """
        self.log_and_print(f'Starting step {self.batch_id}')
        if len(self.dataset.records) == 0:
            self.log_and_print('Dataset is empty generating initial samples')
            self.generate_initial_samples()
        if self.config.use_wandb:
            cur_batch = self.dataset.get_leq(self.batch_id)
            random_subset = cur_batch.sample(n=min(10, len(cur_batch)))[['text']]
            self.wandb_run.log(
                {"Prompt": wandb.Html(f"<p>{self.cur_prompt}</p>"), "Samples": wandb.Table(dataframe=random_subset)},
                step=self.batch_id)

        logging.info('Running annotator')
        records = self.annotator.apply(self.dataset, self.batch_id)
        self.dataset.update(records)

        self.predictor.cur_instruct = self.cur_prompt
        logging.info('Running Predictor')
        records = self.predictor.apply(self.dataset, self.batch_id, leq=True)
        self.dataset.update(records)

        self.eval.dataset = self.dataset.get_leq(self.batch_id)
        self.eval.eval_score()
        logging.info('Calculating Score')
        large_errors = self.eval.extract_errors()
        self.eval.add_history(self.cur_prompt, self.task_description)
        if self.config.use_wandb:
            large_errors = large_errors.sample(n=min(6, len(large_errors)))
            correct_samples = self.eval.extract_correct()
            correct_samples = correct_samples.sample(n=min(6, len(correct_samples)))
            vis_data = pd.concat([large_errors, correct_samples])
            self.wandb_run.log({"score": self.eval.history[-1]['score'],
                                "prediction_result": wandb.Table(dataframe=vis_data),
                                'Total usage': self.calc_usage()}, step=self.batch_id)
        if self.stop_criteria():
            self.log_and_print('Stop criteria reached')
            return True
        if current_iter != total_iter-1:
            self.run_step_prompt()
        self.save_state()
        return False

    def run_pipeline(self, num_steps: int):
        # Run the optimization pipeline for num_steps
        num_steps_remaining = num_steps - self.batch_id
        for i in range(num_steps_remaining):
            stop_criteria = self.step(i, num_steps_remaining)
            if stop_criteria:
                break
        final_result = self.extract_best_prompt()
        return final_result
