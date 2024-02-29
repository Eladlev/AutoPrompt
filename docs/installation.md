# Installation

This guide provides detailed instructions for setting up your development environment, configuring LLMs, and integrating various tools necessary for your project.

## Python version
We recommend using python 3.10.13

## Install with Conda
We recommend installing using Conda:
```bash
conda env create -f environment_dev.yml
conda activate AutoPrompt
```

## Install with pip
Install using pip directly:
```bash
pip install -r requirements.txt
```

## Install with pipenv
Install using pipenv:
```bash
pip install pipenv
pipenv sync
```

### Configure your LLM

Set your OpenAI API key in the configuration file `config/llm_env.yml`. For assistance locating your API key, visit this [link](https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key).

- For LLM, we recommend using [OpenAI's GPT-4](https://platform.openai.com/docs/guides/gpt). Alternatively, configure Azure by setting llm type in `config/config_default.yml` to `"Azure"` and specifying the key in `config/llm_env.yml`. Our system also supports various LLMs, including open source models, through [Langchain Pipeline](https://python.langchain.com/docs/integrations/llms/huggingface_pipelines). Change the llm `type` to `"HuggingFacePipeline"` and specify the model ID in the llm `name` field.  

- **Configure your Predictor**.  We employ a predictor to estimate prompt performance. The default predictor LLM is GPT-3.5. Configuration is located in the `predictor` section of `config/config_default.yml`.

### Configure Human-in-the-Loop Annotator 

Our pipeline incorporates a human-in-the-loop annotation process using [Argilla](https://docs.argilla.io/en/latest/index.html). Follow these steps to set it up:

1. **Set Up Argilla Server and UI:** Follow the [instructions](https://docs.argilla.io/en/latest/getting_started/quickstart_installation.html) to install and set up an Argilla server and user interface.

2. **Quick Installation Option:** For a faster setup, we recommend deploying Argilla on a Hugging Face [space](https://huggingface.co/new-space?template=argilla/argilla-template-space).

3. **Configure API Settings:** After setting up the server, modify the `api_url` and `api_key` in the `config/config_default.yml` file. For instance, if using the recommended Hugging Face space, your API URL should be formatted as follows: `api_url: 'https://<your-argilla-space-name>.hf.space'`.


### Configure LLM Annotator 

To specify an LLM as the annotation tool in your pipeline, update the `annotator` section in the `config/config_default.yml` file as follows:

``` 
annotator:
    method: 'llm'
    config:
        llm:
            type: 'OpenAI'
            name: 'gpt-4-1106-preview'
        instruction:
            'Assess whether the text contains a harmful topic. 
            Answer Yes if it does and No otherwise.'
        num_workers: 5
        prompt: 'prompts/predictor_completion/prediction.prompt'
        mini_batch_size: 1
        mode: 'annotation'
```
We recommend using a robust LLM, like GPT-4, for annotation purposes. In the `instruction` field, you specify the task instructions for the annotation. The `mini_batch_size` field determines the number of samples processed in a single annotation pass, allowing you to balance efficiency with LLM token usage.


### Monitoring: Weights and Biases Setup

To effectively track your optimization process, including metrics like score, prompts instances, and error analysis  across iterations, we recommend using [Weights and Biases](https://wandb.ai/site).

1. **Sign Up for Weights and Biases:** Visit their [website](https://wandb.ai/site) and follow the instructions to create an account.

2. **Enable wandb in Your Configuration:** In your project's `config/config_default.yml` file, set `use_wandb` to `True` to activate wandb support.