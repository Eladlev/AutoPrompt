import yaml
from easydict import EasyDict as edict
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import AzureChatOpenAI

LLM_ENV = yaml.safe_load(open('config/llm_env.yml', 'r'))


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
    if 'temperature' not in config:
        temperature = 0
    else:
        temperature = config['temperature']
    if 'model_kwargs' in config:
        model_kwargs = config['model_kwargs']
    else:
        model_kwargs = {}
    if config['type'] == 'OpenAI':
        if LLM_ENV['openai']['OPENAI_ORGANIZATION'] == '':
            return ChatOpenAI(temperature=temperature, model_name=config['name'],
                              openai_api_key=LLM_ENV['openai']['OPENAI_API_KEY'],
                              model_kwargs=model_kwargs)
        else:
            return ChatOpenAI(temperature=temperature, model_name=config['name'],
                              openai_api_key=LLM_ENV['openai']['OPENAI_API_KEY'],
                              openai_organization=LLM_ENV['openai']['OPENAI_ORGANIZATION'],
                              model_kwargs=model_kwargs)
    elif config['type'] == 'Azure':
        AzureChatOpenAI(temperature=temperature, model_name=config['name'],
                        openai_api_key=LLM_ENV['azure']['AZURE_OPENAI_API_KEY'],
                        azure_endpoint=LLM_ENV['azure']['AZURE_OPENAI_ENDPOINT'],
                        openai_api_version=LLM_ENV['azure']['OPENAI_API_VERSION'])


    elif config['type'] == 'HuggingFacePipeline':
        return HuggingFacePipeline.from_model_id(
            model_id=config['name'],
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": config['max_new_tokens']},
        )
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
    return edict(yaml_data)


def load_prompt(prompt_path: str) -> PromptTemplate:
    """
    Reads and returns the contents of a prompt file.
    :param prompt_path: The path to the prompt file
    :param appendix: A string to append to the prompt
    """
    with open(prompt_path, 'r') as file:
        prompt = file.read().rstrip()
    return PromptTemplate.from_template(prompt)
