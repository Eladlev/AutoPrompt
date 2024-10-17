from optimization_agent import AgentOptimization
from utils.config import modify_input_for_ranker, override_config
import argparse
import os
from estimator.estimator_llm import LLMEstimator
import pickle
import pandas as pd
# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--config_path', default='config/config_diff/config_agent.yml', type=str, help='Configuration file path')

parser.add_argument('--task_description',
                    default='Answer the user query using the external tools.',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--initial_system_prompt',
                    default="",
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--load_dump', default='dump', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--inference_dataset', default='dump/root/dataset.csv',
                    required=False, type=str, help='The csv dataset on which the system will apply the prediction')
opt = parser.parse_args()


config_params = override_config(opt.config_path)
if (opt.load_dump != '') and os.path.isfile(os.path.join(opt.load_dump, 'state.pkl')):  # load the state from the checkpoint
    agent_pipeline = pickle.load(open(os.path.join(opt.load_dump, 'state.pkl'), 'rb'))
else:
    agent_pipeline = AgentOptimization(config_params,  output_path='')

records = pd.read_csv(opt.inference_dataset)
records = agent_pipeline.meta_agent.predict_records(records,task_description = opt.task_description,
                                                    initial_prompt = opt.initial_system_prompt)
records.to_csv(opt.inference_dataset)