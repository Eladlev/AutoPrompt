<p align="center">
    <!-- community badges -->
    <a href="https://discord.gg/G2rSbAf8uP"><img src="https://img.shields.io/badge/Join-Discord-blue.svg"/></a>
    <!-- license badge -->
    <a href="https://github.com/Eladlev/AutoPrompt/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

# 📝 AutoPrompt


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

**Auto Prompt is a prompt optimization framework designed to enhance and perfect your prompts for real-world use cases.**

The framework automatically generates high-quality, detailed prompts tailored to user intentions. It employs a refinement (calibration) process, where it iteratively builds a dataset of challenging edge cases and optimizes the prompt accordingly. This approach not only reduces manual effort in prompt engineering but also effectively addresses common issues such as prompt [sensitivity](https://arxiv.org/abs/2307.09009) and inherent prompt [ambiguity](https://arxiv.org/abs/2311.04205) issues.


**Our mission:** Empower users to produce high-quality robust prompts using the power of large language models (LLMs).

# Why Auto Prompt?
- **Prompt Engineering Challenges.** The quality of LLMs greatly depends on the prompts used. Even [minor changes](#prompt-sensitivity-example) can significantly affect their performance. 
- **Benchmarking Challenges.**  Creating a benchmark for production-grade prompts is often labour-intensive and time-consuming.
- **Reliable Prompts.** Auto Prompt generates robust high-quality prompts, offering measured accuracy and performance enhancement using minimal data and annotation steps.
- **Modularity and Adaptability.** With modularity at its core, Auto Prompt integrates seamlessly with popular open-source tools such as LangChain, Wandb, and Argilla, and can be adapted for a variety of tasks, including data synthesis and prompt migration.

## System Overview

![System Overview](./docs/AutoPrompt_Diagram.png)

The system is designed for real-world scenarios, such as moderation tasks, which are often  challenged by imbalanced data distributions. The system implements the [Intent-based Prompt Calibration](https://arxiv.org/abs/2402.03099) method. The process begins with a user-provided initial prompt and task description, optionally including user examples. The refinement process iteratively generates diverse samples, annotates them via user/LLM, and evaluates prompt performance, after which an LLM suggests an improved prompt.  

The optimization process can be extended to content generation tasks by first devising a ranker prompt and then performing the prompt optimization with this learned ranker. The optimization concludes upon reaching the budget or iteration limit.  


This joint synthetic data generation and prompt optimization approach outperform traditional methods while requiring minimal data and iterations. Learn more in our paper
[Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases](https://arxiv.org/abs/2402.03099) by E. Levi et al. (2024).


**Using GPT-4 Turbo, this optimization typically completes in just a few minutes at a cost of under $1.** To manage costs associated with GPT-4 LLM's token usage, the framework enables users to set a budget limit for optimization, in USD or token count, configured as illustrated [here](docs/examples.md#steps-to-run-example).

## Demo

![pipeline_recording](./docs/autoprompt_recording.gif)


## 📖 Documentation
 - [How to install](docs/installation.md) (Setup instructions)
 - [Prompt optimization examples](docs/examples.md) (Use cases: movie review classification, generation, and chat moderation)
 - [How it works](docs/how-it-works.md) (Explanation of pipelines)
 - [Architecture guide](docs/architecture.md) (Overview of main components)

## Features
- 📝 Boosts prompt quality with a minimal amount of data and annotation steps.
- 🛬 Designed for production use cases like moderation, multi-label classification, and content generation.
- ⚙️ Enables seamless migrating of prompts across model versions or LLM providers.
- 🎓 Supports prompt squeezing. Combine multiple rules into a single efficient prompt.


## QuickStart
AutoPrompt requires `python <= 3.10`
<br />

> **Step 1** - Download the project

```bash
git clone git@github.com:Eladlev/AutoPrompt.git
cd AutoPrompt
```

<br />

> **Step 2** - Install dependencies

Use either Conda or pip, depending on your preference. Using Conda:
```bash
conda env create -f environment_dev.yml
conda activate AutoPrompt
```

Using pip: 
```bash
pip install -r requirements.txt
```

Using pipenv:
```bash
pip install pipenv
pipenv sync
```

<br />

> **Step 3** - Configure your LLM. 

Set your OpenAI API key  by updating the configuration file `config/llm_env.yml`
- If you need help locating your API key, visit this [link](https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key).

- We recommend using [OpenAI's GPT-4](https://platform.openai.com/docs/guides/gpt) for the LLM. Our framework also supports other providers and open-source models, as discussed [here](docs/installation.md#configure-your-llm).

<br />

> **Step 4** - Configure your Annotator
- Select an annotation approach for your project. We recommend beginning with a human-in-the-loop method, utilizing [Argilla](https://docs.argilla.io/en/latest/index.html). Follow the [Argilla setup instructions](docs/installation.md#configure-human-in-the-loop-annotator-) to configure your server. Alternatively, you can set up an LLM as your annotator by following these [configuration steps](docs/installation.md#configure-llm-annotator-).

- The default predictor LLM, GPT-3.5, for estimating prompt performance, is configured in the `predictor` section of `config/config_default.yml`.

- Define your budget in the input config yaml file using the `max_usage parameter`. For OpenAI models, `max_usage` sets the maximum spend in USD. For other LLMs, it limits the maximum token count.

<br />


> **Step 5** - Run the pipeline

First, configure your labels by editing `config/config_default.yml`
```
dataset:
    label_schema: ["Yes", "No"]
```


For a **classification pipeline**, use the following command from your terminal within the appropriate working directory: 
```bash
python run_pipeline.py
```
If the initial prompt and task description are not provided directly as input, you will be guided to provide these details.  Alternatively, specify them as command-line arguments:
```bash
python run_pipeline.py \
    --prompt "Does this movie review contain a spoiler? answer Yes or No" \
    --task_description "Assistant is an expert classifier that will classify a movie review, and let the user know if it contains a spoiler for the reviewed movie or not." \
    --num_steps 30
```
You can track the optimization progress using the [W&B](https://wandb.ai/site) dashboard, with setup instructions available  [here](docs/installation.md#monitoring-weights-and-biases-setup). 


If you are using pipenv, be sure to activate the environment:
``` bash
pipenv shell
python run_pipeline.py  
```
or alternatively prefix your command with `pipenv run`:
```bash
pipenv run python run_pipeline.py 
```

#### Generation pipeline
To run the generation pipeline, use the following example command:
```bash
python run_generation_pipeline.py \
    --prompt "Write a good and comprehensive movie review about a specific movie." \
    --task_description "Assistant is a large language model that is tasked with writing movie reviews."
```
For more information, refer to our [generation task example](docs/examples.md#generating-movie-reviews-generation-task). 

<br />

Enjoy the results. Completion of these steps yields a **refined (calibrated)
prompt** tailored for your task, alongside a **benchmark** featuring challenging samples,
stored in the default `dump` path.



## Tips

- Prompt accuracy may fluctuate during the optimization. To identify the best prompts, we recommend continuous refinement following the initial generation of the benchmark.  Set the number of optimization iterations with `--num_steps` and control sample generation by specifying `max_samples` in the `dataset` section. For instance, setting `max_samples: 50` and `--num_steps 30` limits the benchmark to 50 samples, allowing for 25 additional refinement iterations, assuming 10 samples per iteration.

- The framework supports checkpoints for easy resumption of optimization from the last saved state. It automatically saves the most recent optimization state in a `dump` path. Use `--output_dump` to set this path and `--load_path` to resume from a checkpoint.
- The iterations include multiple calls to the LLM service, with long prompts and requests for a relatively large amount of generated tokens by the LLM. This might take time ~1 minute (especially in the generative tasks), so please be patient.
- If there are some issues with the Argilla server connection/error, try to restart the space.
<!-- 
Meanwhile, the num_initialize_samples and num_generated_samples fields within the meta_prompts section specify the counts for initial and per iteration sample generation, respectively. -->


## Prompt Sensitivity Example
You write a prompt for identifying movie spoilers:
```
Review the content provided and indicate whether it includes any significant plot revelations or critical points that could reveal important elements of the story or its outcome. Respond with "Yes" if it contains such spoilers or critical insights, and "No" if it refrains from unveiling key story elements.
```
This prompt scores 81 on your [benchmark](docs/examples.md#filtering-movie-reviews-with-spoilers-classification-task)  using GPT-4 LLM. Then, you make a minor modification:
```
Review the text and determine if it provides essential revelations or critical details about the story that would constitute a spoiler. Respond with "Yes" for the presence of spoilers, and "No" for their absence.
```
Surprisingly, the second prompt scores 72, representing an 11% drop in accuracy. This illustrates the need for a careful prompt engineering process.

## 🚀 Contributing

Your contributions are greatly appreciated! If you're eager to contribute, kindly refer to our [Contributing Guidelines](docs/contributing.md)) for detailed information.

<!-- For an insight into our future plans, visit our Project Roadmap. -->
If you wish to be a part of our journey, we invite you to connect with us through our [Discord Community](https://discord.gg/G2rSbAf8uP). We're excited to have you onboard! 

## 🛡 Disclaimer

The AutoPrompt project is provided on an "as-is" basis without any guarantees or warranties, expressed or implied. 

Our perspective on the optimization and usage of prompts:

1. The core objective of AutoPrompt is to refine and perfect prompts to achieve high-quality results. This is achieved through an iterative calibration process, which helps in reducing errors and enhancing the performance of LLMs. However, the framework does not guarantee absolute correctness or unbiased results in every instance.

2. AutoPrompt aims to improve the reliability of prompts and mitigate sensitivity issues, but it does not claim to completely eliminate such issues. 
<!-- Our community is committed to exploring the most effective ways to interact with LLMs, fostering a space for diverse views and approaches. -->

Please note that using LLMs like OpenAI's GPT-4, supported by AutoPrompt, may lead to significant costs due to token usage. By using AutoPrompt, you acknowledge your responsibility to monitor and manage your token use and expenses. We advise regularly reviewing your LLM provider's API usage and establishing limits or alerts to prevent unexpected charges.
To manage costs associated with GPT-4 LLM's token usage, the framework enables users to set a budget limit for optimization, in USD or token count, configured as illustrated [here](docs/examples.md#steps-to-run-example).

## Citation

If you have used our code in your research, please cite our [paper](https://arxiv.org/abs/2402.03099):

```
@misc{2402.03099,
Author = {Elad Levi and Eli Brosh and Matan Friedmann},
Title = {Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases},
Year = {2024},
Eprint = {arXiv:2402.03099},
}
```


## License

This framework is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## ✉️ Support / Contact us
- [Community Discord](https://discord.gg/G2rSbAf8uP)
- Our email: [‫autopromptai@gmail.com‬](mailto:autopromptai@gmail.com)


