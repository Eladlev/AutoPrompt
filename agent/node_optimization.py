from utils.config import override_config
from optimization_pipeline import OptimizationPipeline
from agent.agent_utils import get_tools_description
from agent.agent_instantiation import AgentNode, get_var_schema
import os
from easydict import EasyDict as edict
from pathlib import Path


def get_schema_metric(output_schema: str):
    """
    Get the metric that evaluate the adherence to the schema
    """
    prompt = f"""Evaluate the performance of our agent using a five-point assessment scale that measure how well the agent preserve the exact requested ouput structure. 
The agent response should include be a valid YAML structure with the following schema:
{output_schema}
Assign a single score of either 1, 2, 3, 4, or 5, with each level representing different degrees of perfection.
score 1: The response structure is not a valid YAML structure or does not match the required schema.
score 2: The response structure contains illegal YAML syntax or does not match the required schema.
score 3: The response structure is a valid YAML structure but does not match the required schema.
score 4: The response structure is a valid YAML structure and partially matches the required schema.
score 5: The response structure is a valid YAML structure and matches the required schema perfectly."""
    description = "Measuring whether the agent response is a valid YAML structure and matches the required schema."
    return {'metric_name': 'Output structure adherence', 'metric_desc': description, 'metric_prompt': prompt}


def get_parameters_str(input_schema, output_schema):
    """
    Get the part of the prompt that guide the user to provide the correct input and output schema (this
    is for the prompt generation)
    """
    input_schema = get_var_schema(input_schema)
    yaml_schema = get_var_schema(output_schema)
    return f"""The agent has a specific input, and required to generate a YAML output with a given schema.
Therefore the generate prompt should **explicitly** and clearly indicate that the input for the agent is:
{input_schema}
The generate prompt should **explicitly** and clearly indicate that the output structure should be a valid YAML structure with the following structure:
{yaml_schema}"""


def load_predictor_config(config_name: str, config: edict):
    """
    Setup the predictor config for the optimizer
    """
    if config_name == 'agent':
        return {'method': 'agent',
                'config': config.predictor.config}
    else:
        raise NotImplementedError("Predictor not implemented")


def run_agent_optimization(node: AgentNode, output_dump: str,
                           config_base: edict,
                           agent_tools=None,
                           config_path: str = 'config/config_diff/config_generation.yml',
                           num_generation_steps: int = 3):
    """
    Run the agent optimization
    :param node: The agent node
    :param config_base: The base configuration
    :param agent_tools: The available tools
    :param output_dump: The output dump
    :param config_path: The configuration
    :param num_generation_steps: The number of generation steps
    """

    config_params = override_config(config_path)
    predictor_config = load_predictor_config('agent', config_base)
    predictor_config['config']['tools'] = agent_tools
    config_params.metric_generator.metrics = [get_schema_metric(node.function_metadata['outputs'])]
    config_params.predictor = predictor_config
    config_params.meta_prompts.folder = Path('prompts/meta_prompts_agent')
    if config_params.predictor.method == 'agent':
        # node.function_metadata['tools']
        tools_description, tools = get_tools_description(agent_tools)
        initial_prompt = {'prompt': node.function_metadata['prompt'], 'task_tools_description': tools_description}
        task_metadata = {'task_tools_description': tools_description,
                         'tools_names': '\n'.join([tool for tool in tools.keys()]),
                         'additional_instructions': get_parameters_str(node.function_metadata['inputs'],
                                                                       node.function_metadata['outputs'])}

    generation_pipeline = OptimizationPipeline(config_params, node.function_metadata['function_description'],
                                               initial_prompt,
                                               output_path=os.path.join(output_dump, node.function_metadata['name']),
                                               task_metadata=task_metadata)
    best_generation_prompt = generation_pipeline.run_pipeline(num_generation_steps)
    best_generation_prompt['metrics_info'] = generation_pipeline.metrics_info
    node.function_metadata['prompt'] = best_generation_prompt['prompt']['prompt']
    node.function_metadata['tools_metadata'] = best_generation_prompt['prompt']['tools_metadata']
    return best_generation_prompt
