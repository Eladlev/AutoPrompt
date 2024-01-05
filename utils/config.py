import yaml
from easydict import EasyDict as edict
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
import logging

LLM_ENV = yaml.safe_load(open('config/llm_env.yml', 'r'))

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'  # Reset to default color


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
        yaml_data['meta_prompts']['folder'] = Path(yaml_data['meta_prompts']['folder'])
    return edict(yaml_data)


def load_prompt(prompt_path: str) -> PromptTemplate:
    """
    Reads and returns the contents of a prompt file.
    :param prompt_path: The path to the prompt file
    """
    with open(prompt_path, 'r') as file:
        prompt = file.read().rstrip()
    return PromptTemplate.from_template(prompt)


def validate_generation_config(base_config, generation_config):
    if "estimator" not in generation_config:
        raise Exception("Generation config must contain an empty estimator.")
    if "label_schema" not in generation_config.dataset or \
            base_config.dataset.label_schema != generation_config.dataset.label_schema:
        raise Exception("Generation label schema must match the basic config.")


def modify_input_for_ranker(config, task_description, initial_prompt):
    modifiers_config = yaml.safe_load(open('prompts/modifiers/modifiers.yml', 'r'))
    task_desc_setup = load_prompt(modifiers_config['ranker']['task_desc_mod'])
    init_prompt_setup = load_prompt(modifiers_config['ranker']['prompt_mod'])

    llm = get_llm(config.llm)
    task_llm_chain = LLMChain(llm=llm, prompt=task_desc_setup)
    task_result = task_llm_chain(
        {"task_description": task_description})
    mod_task_desc = task_result['text']
    logging.info(f"Task description modified for ranking to: \n{mod_task_desc}")

    prompt_llm_chain = LLMChain(llm=llm, prompt=init_prompt_setup)
    prompt_result = prompt_llm_chain({"prompt": initial_prompt, 'label_schema': config.dataset.label_schema})
    mod_prompt = prompt_result['text']
    logging.info(f"Initial prompt modified for ranking to: \n{mod_prompt}")

    return mod_prompt, mod_task_desc
