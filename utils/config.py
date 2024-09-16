import os.path

import yaml
from easydict import EasyDict as edict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_models import AzureChatOpenAI

from langchain.chains import LLMChain
import logging

from openai import OpenAI
import base64
import os
import requests
import asyncio

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

    if config['type'].lower() == 'openai':
        if LLM_ENV['openai']['OPENAI_ORGANIZATION'] == '':
            openai_organization = config.get('openai_organization', LLM_ENV['openai']['OPENAI_ORGANIZATION'])
        else:
            openai_organization = None
        return ChatOpenAI(temperature=temperature, model_name=config['name'],
                          openai_api_key=config.get('openai_api_key', LLM_ENV['openai']['OPENAI_API_KEY']),
                          openai_api_base=config.get('openai_api_base', 'https://api.openai.com/v1'),
                          openai_organization=openai_organization,
                          model_kwargs=model_kwargs)

    elif config['type'].lower() == 'azure':
        return AzureChatOpenAI(temperature=temperature, azure_deployment=config['name'],
                               openai_api_key=config.get('openai_api_key', LLM_ENV['azure']['AZURE_OPENAI_API_KEY']),
                               azure_endpoint=config.get('azure_endpoint', LLM_ENV['azure']['AZURE_OPENAI_ENDPOINT']),
                               openai_api_version=config.get('openai_api_version',
                                                             LLM_ENV['azure']['OPENAI_API_VERSION']))

    elif config['type'].lower() == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(temperature=temperature, model=config['name'],
                                      google_api_key=LLM_ENV['google']['GOOGLE_API_KEY'],
                                      model_kwargs=model_kwargs)

    elif config['type'].lower() == 'huggingfacepipeline':
        device = config.get('gpu_device', -1)
        device_map = config.get('device_map', None)

        return HuggingFacePipeline.from_model_id(
            model_id=config['name'],
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": config['max_new_tokens']},
            device=device,
            device_map=device_map
        )

    else:
        raise NotImplementedError("LLM not implemented")


def get_t2i_model(config: dict):
    if config['type'].lower() == 'openai':
        client = OpenAI(api_key=LLM_ENV['openai']['OPENAI_API_KEY'])

        def generate_single_image(prompt):
            response = client.images.generate(
                model=config['name'],
                prompt=prompt,
                size=config['image_size'],
                quality=config['quality'],
                n=1,
            )
            img_url = [im.url for im in response.data]
            return img_url

        async def generate_images_async(prompt, num_images=1):
            response = client.images.generate(
                model=config['name'],
                prompt=prompt,
                size=config['image_size'],
                quality=config['quality'],
                n=num_images,
            )
            return response

        async def run_image_generation_batch(prompts):
            tasks = [generate_images_async(prompt, num_images=1) for prompt in prompts]
            responses = await asyncio.gather(*tasks)
            url_list = [im.url for response in responses for im in response.data]
            revised_prompts = [im.revised_prompt for response in responses for im in response.data]
            return url_list

        def generate_images(prompt, num_images=1):
            if num_images == 1:
                img_urls = generate_single_image(prompt)
            else:
                prompts = [prompt]*num_images
                img_urls = asyncio.run(run_image_generation_batch(prompts))
            return img_urls

        return generate_images
    
    elif config['type'].lower() == 'stability':
        api_key = LLM_ENV['stability']["STABILITY_API_KEY"]
        api_host = LLM_ENV['stability'].get('API_HOST', 'https://api.stability.ai')

        if config['name'] in ['stable-diffusion-xl-1024-v1-0', 'stable-diffusion-v1-6']:
            engine_id = config['name']

            def generate_images(prompt, num_images=1):
                if api_key is None:
                    raise Exception("Missing Stability API key.")

                response = requests.post(
                    f"{api_host}/v1/generation/{engine_id}/text-to-image",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    },
                    json={
                        "text_prompts": [
                            {
                                "text": prompt
                            }
                        ],
                        "cfg_scale": 7,
                        "height": config['image_size'][1],
                        "width": config['image_size'][0],
                        "samples": num_images,
                        "steps": 30,
                    },
                )

                if response.status_code != 200:
                    raise Exception("Non-200 response: " + str(response.text))

                data = response.json()
                files_location = []
                im_num = len(os.listdir(config['output_path']))
                for i, image in enumerate(data["artifacts"]):
                    fn = f"{config['output_path']}/{im_num}_v1_txt2img_{i}.png"
                    files_location.append(fn)
                    with open(fn, "wb") as f:
                        f.write(base64.b64decode(image["base64"]))
                return files_location

            return generate_images
        elif config['name'] in ['ultra', 'core', 'sd3']:

            def generate_images(prompt, num_images=1):
                response = requests.post(
                    f"https://api.stability.ai/v2beta/stable-image/generate/{config['name']}",
                    headers={
                        "authorization": f"Bearer {api_key}",
                        "accept": "image/*"
                    },
                    files={"none": ''},
                    data={
                        "prompt": prompt,
                        "output_format": "png",
                    },
                )

                im_num = len(os.listdir(config['output_path']))
                fn = f"{config['output_path']}/{im_num}_v1_txt2img_{0}.png"

                if response.status_code == 200:
                    with open(fn, 'wb') as file:
                        file.write(response.content)
                else:
                    raise Exception(str(response.json()))
                return [fn]
            return generate_images
        else:
            raise NotImplementedError("Stability model not implemented")


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


def load_prompt(prompt_input: str) -> PromptTemplate:
    """
    Reads and returns the contents of a prompt file.
    :param prompt_input: Either The path to the prompt file or the prompt itself
    """
    if os.path.isfile(prompt_input):
        with open(prompt_input, 'r') as file:
            prompt = file.read().rstrip()
    else:
        prompt = prompt_input
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
