import pandas as pd
from utils.eval import Eval
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
    2. estimator - The estimator is responsible generate the GT
    3. predictor - The predictor is responsible to generate the prediction
    4. eval - The eval is responsible to calculate the score and the large errors
    """

    def __init__(self, config, task_description: str, initial_prompt: str, output_path: str = ''):
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
        self.estimator = give_estimator(config.estimator)
        self.eval = Eval(config.eval)
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
        total_usage += self.estimator.calc_usage()
        total_usage += self.predictor.calc_usage()
        return total_usage

    def run_step_prompt(self):
        """
        Run the meta-prompts and get new prompt suggestion, estimated prompt score and a set of challenging samples
        for the new prompts
        """
        last_history = self.eval.history[-self.config.meta_prompts.history_length:]
        history_prompt = '\n'.join([Eval.sample_to_text(sample,
                                                        num_errors_per_label=self.config.meta_prompts.num_err_prompt,
                                                        is_score=True) for sample in last_history])

        prompt_suggestion = self.meta_chain.step_prompt_chain.invoke({"history": history_prompt,
                                                                      "task_description": self.task_description,
                                                                      "labels": json.dumps(
                                                                          self.config.dataset.label_schema)})

        history_samples = '\n'.join([Eval.sample_to_text(sample,
                                                         num_errors_per_label=self.config.meta_prompts.num_err_samples,
                                                         is_score=False) for sample in last_history])

        self.log_and_print(f'Get new prompt:\n{prompt_suggestion["prompt"]}')
        self.batch_id += 1
        if len(self.dataset) < self.config.dataset.max_samples:
            new_samples = self.meta_chain.step_samples.invoke({"history": history_samples,
                                                               "task_description": self.task_description,
                                                               "prompt": prompt_suggestion['prompt'],
                                                               'num_samples':
                                                                   self.config.meta_prompts.num_generated_samples})
            self.dataset.add(new_samples['samples'], self.batch_id)
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
        min_batch_id, min_score = self.eval.get_min_score()
        if self.eval.history[-1]['score'] - min_score > self.config.stop_criteria.min_delta:
            self.patient += 1
        else:
            self.patient = 0
        if self.patient > self.config.stop_criteria.patience:
            return True
        return False

    def generate_initial_samples(self):
        """
        In case the initial dataset is empty generate the initial samples
        """
        samples_list = self.meta_chain.initial_chain.invoke(
            {"sample_number": self.config.dataset.num_initialize_samples,
             "task_description": self.task_description,
             "instruction": self.cur_prompt})
        self.dataset.add(samples_list['samples'], 0)

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
        self.dataset.load_dataset(path / 'dataset.csv')
        state = pickle.load(open(path / 'history.pkl', 'rb'))
        self.eval.history = state['history']
        self.batch_id = state['batch_id']
        self.cur_prompt = state['prompt']
        self.task_description = state['task_description']
        self.patient = state['patient']

    def step(self):
        """
        This is the main optimization process step.
        """
        self.log_and_print(f'Starting step {self.batch_id}')
        if len(self.dataset.records) == 0:
            logging.info('Dataset is empty generating initial samples')
            self.generate_initial_samples()
        if self.config.use_wandb:
            cur_batch = self.dataset[self.batch_id]
            random_subset = cur_batch.sample(n=min(10, len(cur_batch)))[['text']]
            self.wandb_run.log({"Prompt":  wandb.Html(f"<p>{self.cur_prompt}</p>"), "Samples": wandb.Table(dataframe=random_subset)},
                               step=self.batch_id)

        logging.info('Running Estimator')
        records = self.estimator.apply(self.dataset, self.batch_id)
        self.dataset.update(records)

        self.predictor.cur_instruct = self.cur_prompt
        logging.info('Running Predictor')
        records = self.predictor.apply(self.dataset, self.batch_id, leq=True)
        self.dataset.update(records)

        self.eval.eval_score(self.dataset)
        logging.info('Calculating Score')
        self.eval.dataset = self.dataset.records
        large_errors = self.eval.extract_errors()
        self.eval.add_history(self.cur_prompt)
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
            return self.cur_prompt
        self.run_step_prompt()
        self.save_state()

    def run_pipeline(self, num_steps: int):
        # Run the optimization pipeline for num_steps
        num_steps_remaining = num_steps - self.batch_id
        for i in range(num_steps_remaining):
            self.step()
        # TODO: Need to change the cur_prompt to best_prompt
        return self.cur_prompt

    def get_predictor(self):
        # TODO: Need to change the cur_prompt to best_prompt
        return self.predictor

    def set_predictor(self, predictor):
        predictor_score_func = lambda record: Eval.ranker_score_func(record, predictor)
        self.eval.score_func = predictor_score_func

