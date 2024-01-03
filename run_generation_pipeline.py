from optimization_pipeline import OptimizationPipeline
from utils.config import load_yaml, modify_input_for_ranker, validate_generation_config
import argparse
import os
# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--generation_config_path', default='config/config_generation.yml', type=str, help='Configuration file path')
parser.add_argument('--basic_config_path', default='config/config.yml', type=str, help='Configuration file path')
parser.add_argument('--task_description',
                    default='Assistant is a large language model which with the task to write movie reviews.',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--prompt',
                    default='Generate a good movie review.',
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--load_dump', default='dump', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_steps', default=2, type=int, help='Number of iterations')

opt = parser.parse_args()

generation_config_params = load_yaml(opt.generation_config_path)
base_config_params = load_yaml(opt.basic_config_path) #TODO: change it to diff yaml
validate_generation_config(base_config_params, generation_config_params)

if opt.task_description == '':
    task_description = input("Describe the task: ")
else:
    task_description = opt.task_description

if opt.prompt == '':
    initial_prompt = input("Initial prompt: ")
else:
    initial_prompt = opt.prompt

ranker_pipeline = OptimizationPipeline(base_config_params, output_path=os.path.join(opt.output_dump, 'ranker'))
if opt.load_dump != '':
    ranker_pipeline.load_state(os.path.join(opt.load_dump, 'ranker'))
    if (ranker_pipeline.cur_prompt is None) or (ranker_pipeline.task_description is None):
        ranker_mod_prompt, ranker_mod_task_desc = modify_input_for_ranker(base_config_params, task_description,
                                                                          initial_prompt)
        ranker_pipeline.cur_prompt = ranker_mod_prompt
        ranker_pipeline.task_description = ranker_mod_task_desc

best_prompt = ranker_pipeline.run_pipeline(opt.num_steps)
predictor = ranker_pipeline.get_predictor()

generation_pipeline = OptimizationPipeline(generation_config_params, task_description, initial_prompt,
                                           output_path=os.path.join(opt.output_dump, 'generator'))
if opt.load_dump != '':
    generation_pipeline.load_state(os.path.join(opt.load_dump, 'generator'))
generation_pipeline.set_predictor(predictor)
best_generation_prompt = generation_pipeline.run_pipeline(opt.num_steps)
