import openai
from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import os
import sys
import yaml
sys.path.append('./../utils/')
sys.path.append('./../')
from config import load_yaml
from llm_chain import ChainWrapper
import json

class MetricScale(BaseModel):
    deficient_desc: str = Field(description="1. Deficient Performance on (a few words about the metric go here) (1): a single sentence description of what should be checked in the task output such that it classifies as a Deficient level of performance relative to this metric.")
    adequate_desc: str = Field(description="2. Adequate Performance on (a few words about the metric go here) (2): a single sentence description of what should be checked in the task output such that it classifies as a Adequate level of performance relative to this metric.")
    competent_desc: str = Field(description="3. Competent Performance on (a few words about the metric go here) (3): a single sentence description of what should be checked in the task output such that it classifies as a Competent level of performance relative to this metric.")
    proficient_desc: str = Field(description="4. Proficient Performance on (a few words about the metric go here) (4): a single sentence description of what should be checked in the task output such that it classifies as a Proficient level of performance relative to this metric.")
    exemplary_desc: str = Field(description="5. Exemplary Performance on (a few words about the metric go here) (5): a single sentence description of what should be checked in the task output such that it classifies as a Exemplary level of performance relative to this metric.") 

class Metric(BaseModel):
    metric_name: str = Field(description="The name of the evaluation metric, in a few words, that will serve as the area of diagnostic evaluation for the task.")
    metric_desc: str = Field(description="Explanation, in not more than 3 sentences, what the above metric means and what it's trying to assess.")
    metric_prompt: str = Field(description="A prompt that will be used as input to an external evaluator agent to evaluate the assistant's performance based on the metric name and metric description generated above. This prompt should start with the phrase: 'Evaluate the performance of our agent using a five-point assessment scale that emphasizes', followed by a few words about the metric name and description, followed by another sentence starting with the phrase: 'Assign a single score of either 1, 2, 3, 4, or 5, with each level representing different degrees of perfection with respect to the', followed by just a few words again summarizing the metric name and description.")
    metric_scale: List[MetricScale] = Field(description="A list of descriptions for each scale of assessment from 1 to 5 for the given metric. Note that in the above metric prompt structure, 1 represents the lowest level of performance and 5 represents the best level of performance.")

class MetricGeneratorFlow(BaseModel):
    metrics_list: List[Metric] = Field(description="The list of all possible rules that would be important to assess whether this assistant performed the above task perfectly or not.")

config_params = load_yaml('./config_metric_gen/config_llm.yml')
llm_config = config_params['llm']

prompt_path = './meta_prompts_metric_gen/metric_gen_rag_qa.prompt'

chain = ChainWrapper(llm_config, prompt_path, MetricGeneratorFlow, None)

res = chain.invoke({'task_description': "Assistant is a large language model that is tasked to provide answers to questions after retrieving information from a given knowledge base.",
                'instruction': "You will be given a location to a knowledge base of documents covering music history. You need to answer the following questions about musicians as accurately as possible."})

res_json = res.json(indent=4)
with open('metrics.json', 'w') as json_file:
    json_file.write(res_json)
