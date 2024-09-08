from optimization_agent import AgentOptimization
from utils.config import modify_input_for_ranker, override_config
import argparse
import os
from estimator.estimator_llm import LLMEstimator
import pickle
# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--config_path', default='config/config_diff/config_agent.yml', type=str, help='Configuration file path')

parser.add_argument('--task_description',
                    default="",
                    required=False, type=str, help='Describing the agent task')
parser.add_argument('--initial_system_prompt',
                    default="",
                    required=False, type=str, help='Initial system prompt, can be either text or a path to a file')
parser.add_argument('--load_dump', default='dump', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')

opt = parser.parse_args()


config_params = override_config(opt.config_path)
if (opt.load_dump != '') and os.path.isfile(os.path.join(opt.load_dump, 'state.pkl')):  # load the state from the checkpoint
    agent_pipeline = pickle.load(open(os.path.join(opt.load_dump, 'state.pkl'), 'rb'))
else:
    agent_pipeline = AgentOptimization(config_params,  output_path=opt.output_dump)
    if opt.task_description == '' and opt.initial_system_prompt == '':
        task_description = input("Please provide either a task description or an agent system prompt\nTask description: ")

agent_pipeline.optimize_agent(opt.task_description, opt.initial_system_prompt)
