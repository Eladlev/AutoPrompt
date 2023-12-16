from optimization_pipeline import OptimizationPipeline
from utils.config import load_yaml
import argparse

# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--config_path', default='config/config_generation_matan1.yml', type=str, help='Configuration file path')
parser.add_argument('--prompt',
                    default='Compose a short poem about the beauty of nature.',
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--task_description',
                    default='you are an amazing award winning poet. you write articulated poems.',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--load_ranker_path', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--load_generator_path', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_steps', default=2, type=int, help='Number of iterations')

opt = parser.parse_args()

config_params = load_yaml(opt.config_path)
if opt.task_description == '':
    task_description = input("Describe the task: ")
else:
    task_description = opt.task_description

if opt.prompt == '':
    initial_prompt = input("Initial prompt: ")
else:
    initial_prompt = opt.prompt

ranker_pipeline = OptimizationPipeline(config_params, task_description, initial_prompt, output_path=opt.output_dump, ranker_run=True)
if opt.load_ranker_path != '':
    ranker_pipeline.load_state(opt.load_ranker_path)
ranker = ranker_pipeline.run_pipeline(opt.num_steps, return_predictor=True)

generation_pipeline = OptimizationPipeline(config_params, task_description, initial_prompt, output_path=opt.output_dump)
if opt.load_generator_path != '':
    generation_pipeline.load_state(opt.load_generator_path)
generation_pipeline.run_pipeline(opt.num_steps)

