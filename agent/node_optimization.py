import copy
from utils.config import override_config
from optimization_pipeline import OptimizationPipeline
from agent.agent_utils import get_tools_description, batch_invoke
from agent.agent_instantiation import AgentNode, get_var_schema
import os
from easydict import EasyDict as edict
from pathlib import Path
from utils.llm_chain import dict_to_prompt_text, get_dummy_callback


def get_schema_metric(output_schema: str):
    """
    Get the metric that evaluate the adherence to the schema
    """
    prompt = f"""Evaluate the performance of our agent using a five-point assessment scale that measure how well the agent preserve the exact requested ouput structure. 
The agent response should include be a valid YAML structure with the following structure ```yaml <yaml file> ```:
where the YAML file should contain the following schema:
{output_schema}

Assign a single score of either 1, 2, 3, 4, or 5, with each level representing different degrees of perfection.
score 1: The response structure is not a valid YAML structure or does not match the required schema.
score 2: The response structure contains illegal YAML syntax or does not match the required format```yaml <yaml file> ```.
score 3: The response structure is a valid YAML structure but does not match the required format ```yaml <yaml file> ```.
score 4: The response structure is a valid YAML structure and partially matches the required schema.
score 5: The response structure is a valid YAML structure and matches the required schema perfectly."""
    description = "Measuring whether the agent response is a valid YAML structure and matches the required ```yaml <yaml file> ```"
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
The agent output structure should be a valid YAML structure with the following structure:
{yaml_schema}
The result system prompt must explicitly guide the model to call the 'parse_yaml_code' function before returning the final answer, with a valid YAML code, containing the YAML output!"""


def load_predictor_config(config_name: str, config: edict):
    """
    Setup the predictor config for the optimizer
    """
    if config_name == 'agent':
        return {'method': 'agent',
                'config': config.predictor.config}
    else:
        raise NotImplementedError("Predictor not implemented")


def init_optimization(node: AgentNode, output_dump: str,
                      config_base: edict,
                      agent_tools=None,
                      config_path: str = 'config/config_diff/config_generation.yml',
                      init_metrics: bool = True):
    """
    Initialize the optimization
    :param node: The agent node
    :param config_base: The base configuration
    :param agent_tools: The available tools
    :param output_dump: The output dump
    :param config_path: The configuration
    :param init_metrics: Whether to initialize the metrics
    """
    config_params = override_config(config_path)
    config_base = copy.deepcopy(config_base)
    predictor_config = load_predictor_config('agent', config_base)
    predictor_config['config']['tools'] = agent_tools
    # config_params.metric_generator.metrics = [get_schema_metric(node.function_metadata['outputs'])]
    config_params.metric_generator.metrics = []
    config_params.predictor = predictor_config
    config_params.meta_prompts.folder = Path('prompts/meta_prompts_agent')
    config_params.metric_generator['init_metrics'] = init_metrics
    available_tools = [tool for tool in agent_tools if not tool.name == 'parse_yaml_code']
    tools_description, tools = get_tools_description(available_tools)
    initial_prompt = {'prompt': node.function_metadata['prompt'], 'task_tools_description': tools_description}
    task_metadata = {'task_tools_description': tools_description,
                     'tools_names': '\n'.join([tool for tool in tools.keys()]),
                     'additional_instructions': get_parameters_str(node.function_metadata['inputs'],
                                                                   node.function_metadata['outputs'])}

    generation_pipeline = OptimizationPipeline(config_params, node.function_metadata['function_description'],
                                               initial_prompt,
                                               output_path=os.path.join(output_dump, node.function_metadata['name']),
                                               task_metadata=task_metadata)
    return generation_pipeline


def run_exec(params: dict):
    """
    Run the execution string
    :param params: input to the function
    """
    exec_string = 'prediction = ' + params['input']
    local_scope = params['local_scope']
    exec(exec_string, local_scope)
    return local_scope['prediction']


def run_agent_optimization(node: AgentNode, output_dump: str,
                           config_base: edict,
                           agent_tools=None,
                           config_path: str = 'config/config_diff/config_generation.yml',
                           num_optimization_steps: int = 2):
    """
    Run the agent optimization
    :param node: The agent node
    :param config_base: The base configuration
    :param agent_tools: The available tools
    :param output_dump: The output dump
    :param config_path: The configuration
    :param num_optimization_steps: The number of generation steps
    """

    generation_pipeline = init_optimization(node, output_dump, config_base, agent_tools, config_path)
    generation_pipeline.load_state(os.path.join(output_dump, node.function_metadata['name'])) #Load the state if it exists
    best_generation_prompt = generation_pipeline.run_pipeline(num_optimization_steps)
    best_generation_prompt['metrics_info'] = generation_pipeline.metrics_info
    node.function_metadata['prompt'] = best_generation_prompt['prompt']['prompt']
    if 'tools_metadata' in best_generation_prompt['prompt']:
        node.function_metadata['tools_metadata'] = best_generation_prompt['prompt']['tools_metadata']
    return best_generation_prompt


def run_flow_optimization(node: AgentNode, dump_root_path: str,
                          config_base: edict = None,
                          agent_tools=None,
                          config_path: str = 'config/config_diff/config_generation.yml',
                          num_optimization_steps: int = 2):
    """
    Run the flow optimization
    :param node: The agent node
    :param config_path: The configuration
    :param dump_root_path: The dump root path
    :param config_base: The base configuration
    :param agent_tools: The available tools
    :param num_optimization_steps: The number of optimization steps
    """
    generation_pipeline = init_optimization(node, dump_root_path, config_base,
                                            agent_tools, config_path, init_metrics=False)
    generation_pipeline.load_state(os.path.join(dump_root_path, node.function_metadata['name']), retrain=True)
    best_generation_prompt = generation_pipeline.run_pipeline(num_optimization_steps)
    best_generation_prompt['metrics_info'] = generation_pipeline.metrics_info
    node.function_metadata['prompt'] = best_generation_prompt['prompt']['prompt']
    if 'tools_metadata' in best_generation_prompt['prompt']:
        node.function_metadata['tools_metadata'] = best_generation_prompt['prompt']['tools_metadata']
    return best_generation_prompt
