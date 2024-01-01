from optimization_pipeline import OptimizationPipeline
from utils.config import load_yaml
import argparse

# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--config_path', default='config/config.yml', type=str, help='Configuration file path')
parser.add_argument('--prompt',
                    default='',
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--task_description',
                    default='',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--load_path', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_steps', default=10, type=int, help='Number of iterations')

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

# Initializing the pipeline
pipeline = OptimizationPipeline(config_params, task_description, initial_prompt, output_path=opt.output_dump)
if opt.load_path != '':
    pipeline.load_state(opt.load_path)

# Run the optimization pipeline for num_steps
num_steps = opt.num_steps - pipeline.batch_id
for i in range(num_steps):
    stop_criteria = pipeline.step()
    if stop_criteria:
        break



