import yaml
from easydict import EasyDict as edict
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from utils.output_scehmes import classification_prediction_schema
import json


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'  # Reset to default color


def get_eval_function(function_name: str):
    """
    Returns the eval function
    :param function_name: The function name
    :return: The function implementation on a record
    """
    if function_name == 'accuracy':
        return lambda record: record['annotation'] == record['prediction']
    else:
        raise NotImplementedError("Eval function not implemented")


def get_llm(config: dict):
    """
    Returns the LLM model
    :param config: dictionary with the configuration
    :return: The llm model
    """
    if config['type'] == 'OpenAI':
        return ChatOpenAI(temperature=0, model_name=config['name'])
    elif config['type'] == 'HuggingFaceHub':
        # TODO: add from here https://github.com/noamgat/lm-format-enforcer
        return NotImplementedError("LLM not implemented")
    else:
        raise NotImplementedError("LLM not implemented")


def load_yaml(yaml_path: str) -> edict:
    """
    Reads the yaml file and enrich it with more vales.
    :param yaml_path: The path to the yaml file
    :return: An EasyDict configuration
    """
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        yaml_data['eval']['score_function'] = get_eval_function(yaml_data['eval']['function_name'])
        yaml_data['meta_prompts']['folder'] = Path(yaml_data['meta_prompts']['folder'])
        classification_prediction_schema['$defs']['Result']['properties']['prediction']['enum'] = yaml_data['dataset'][
            'label_schema']
        classification_prediction_schema['$defs']['Result']['properties']['prediction'][
            'description'] += 'The answer must be one of the following options: {} !!'.format(
            yaml_data['dataset']['label_schema'])
        yaml_data['predictor']['config']['json_schema'] = json.dumps(classification_prediction_schema)
    return edict(yaml_data)


def load_prompt(prompt_path: str, appendix: str = None) -> PromptTemplate:
    """
    Reads and returns the contents of a prompt file.
    :param prompt_path: The path to the prompt file
    :param appendix: A string to append to the prompt
    """
    with open(prompt_path, 'r') as file:
        prompt = file.read().rstrip()
    if appendix is not None:
        prompt += appendix
    return PromptTemplate.from_template(prompt)
