from optimization_pipeline import OptimizationPipeline
from utils.config import override_config
import argparse
# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--config_path', default='config/config_diff/config_images.yml',
                    type=str, help='Configuration file path')
parser.add_argument('--prompt',
                    default='Generate an 8-bit style image of a golden retriever dog playing with a stick near a lake in summer',
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--load_path', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_steps', default=4, type=int, help='Number of iterations')

opt = parser.parse_args()
config_params = override_config(opt.config_path)

if opt.prompt == '':
    initial_prompt = input("Initial prompt: ")
else:
    initial_prompt = opt.prompt
task_description = initial_prompt
config_params.eval['task_description'] = task_description
# Initializing the pipeline
pipeline = OptimizationPipeline(config_params, task_description, initial_prompt, output_path=opt.output_dump)
if opt.load_path != '':
    pipeline.load_state(opt.load_path)
best_prompt = pipeline.run_pipeline(opt.num_steps)
print('\033[92m' + 'Calibrated prompt score:', str(best_prompt['score']) + '\033[0m')
print('\033[92m' + 'Calibrated prompt:', best_prompt['prompt']['prompt'] + '\033[0m')
