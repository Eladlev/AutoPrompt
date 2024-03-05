import yaml
from easydict import EasyDict as edict
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from pathlib import Path
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_models import AzureChatOpenAI
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
        return AzureChatOpenAI(temperature=temperature, deployment_name=config['name'],
                        openai_api_key=LLM_ENV['azure']['AZURE_OPENAI_API_KEY'],
                        azure_endpoint=LLM_ENV['azure']['AZURE_OPENAI_ENDPOINT'],
                        openai_api_version=LLM_ENV['azure']['OPENAI_API_VERSION'])

    elif config['type'] == 'Google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(temperature=temperature, model=config['name'],
                              google_api_key=LLM_ENV['google']['GOOGLE_API_KEY'],
                              model_kwargs=model_kwargs)


    elif config['type'] == 'HuggingFacePipeline':
        return HuggingFacePipeline.from_model_id(
            model_id=config['name'],
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": config['max_new_tokens']},
        )
    else:
        raise NotImplementedError("LLM not implemented")


def load_yaml(yaml_path: str, as_edict: bool = True) -> edict:
    """
    Reads the yaml file and enrich it with more vales.
    :param yaml_path: The path to the yaml file
    :param as_edict: If True, returns an EasyDict configuration
    :return: An EasyDict configuration
    """
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        if 'meta_prompts' in yaml_data.keys() and 'folder' in yaml_data['meta_prompts']:
            yaml_data['meta_prompts']['folder'] = Path(yaml_data['meta_prompts']['folder'])
    if as_edict:
        yaml_data = edict(yaml_data)
    return yaml_data


def load_prompt(prompt_path: str) -> PromptTemplate:
    """
    Reads and returns the contents of a prompt file.
    :param prompt_path: The path to the prompt file
    """
    with open(prompt_path, 'r') as file:
        prompt = file.read().rstrip()
    return PromptTemplate.from_template(prompt)


def validate_generation_config(base_config, generation_config):
    if "annotator" not in generation_config:
        raise Exception("Generation config must contain an empty annotator.")
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


def override_config(override_config_file, config_file='config/config_default.yml'):
    """
    Override the default configuration file with the override configuration file
    :param config_file: The default configuration file
    :param override_config_file: The override configuration file
    """

    def override_dict(config_dict, override_config_dict):
        for key, value in override_config_dict.items():
            if isinstance(value, dict):
                if key not in config_dict:
                    config_dict[key] = value
                else:
                    override_dict(config_dict[key], value)
            else:
                config_dict[key] = value
        return config_dict

    default_config_dict = load_yaml(config_file, as_edict=False)
    override_config_dict = load_yaml(override_config_file, as_edict=False)
    config_dict = override_dict(default_config_dict, override_config_dict)
    return edict(config_dict)
