# A file containing the json schema for the output of all the LLM chains
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Deque, List, Optional, Tuple
from agent.agent_instantiation import Variable
from utils.llm_chain import dict_to_prompt_text


class SubFunction(BaseModel):
    function_name: str = Field(description="The subtask function name")
    function_description: str = Field(description="The subtask function description")
    input_variables: List[Variable] = Field(description="input variables definition")
    output_variables: List[Variable] = Field(description="output variables definition")
    tools_list: List[str] = Field(description="A list of tools names that are used by the subtask function. You can only provide tools from the provided list!!")

class FlowDecomposition(BaseModel):
  # Decomposing the task flow
  sub_functions_list: List[SubFunction] = Field(description="The list of the required subfunctions that are needed to decompose the task")
  code_flow: str = Field(description="A Python code that using the sub functions list compose them to solve the task")



initial_schema = step_samples_schema = {
    "description": "A List of all results",
    "properties": {
        "samples": {
            "description": "Each sample is a string containing the sample content, without any additional information like the Prediction or GT",
            "items": {
                "type": "string"
            },
            "title": "Samples",
            "type": "array"
        }
    },
    "required": [
        "samples"
    ],
    "title": "Sample_List",
    "type": "object"
}

classification_prediction_schema = {
    "$defs": {
        "Result": {
            "description": "A single result",
            "properties": {
                "id": {
                    "description": "The sample id",
                    "title": "Id",
                    "type": "integer"
                },
                "prediction": {
                    "description": "The prediction of the sample.",
                    "title": "Prediction",
                    "type": "string"
                }
            },
            "required": [
                "id",
                "prediction"
            ],
            "title": "Result",
            "type": "object"
        }
    },
    "description": "A List of task classification results",
    "properties": {
        "results": {
            "description": "Each item contain the id and the prediction of the sample",
            "items": {
                "$ref": "#/$defs/Result"
            },
            "title": "Results",
            "type": "array"
        }
    },
    "required": [
        "results"
    ],
    "title": "Results_List",
    "type": "object"
}

step_prompt_schema = {"title": "Suggested_Prompt", "type": "object", "properties": {
    "prompt": {"title": "Prompt", "description": "The agent system prompt description", "type": "string"},
    "tools_description": {"title": "Tools Description", "description": "The tools description", "type": "array",
                          "items": {"$ref": "#/definitions/Tool_Prompt"}}}, "required": ["prompt", "tools_description"],
                      "definitions": {"Tool_Prompt": {"title": "Tool_Prompt", "type": "object", "properties": {
                          "tool_name": {"title": "Tool Name", "description": "The tool name", "type": "string"},
                          "description": {"title": "Description", "description": "The tool description",
                                          "type": "string"}}, "required": ["tool_name", "description"]}}}

breaking_flow_schema = FlowDecomposition

initial_prompt_schema = {
    "description": "An agent system prompt",
    "properties": {
        "prompt": {
            "description": "The result agent system prompt",
            "title": "system prompt",
            "type": "string"
        },
    },
    "required": [
        "prompt",
    ],
    "title": "suggested_prompt",
    "type": "object"
}

build_agent_init_schema = initial_prompt_schema

initial_task_description_schema = {
    "description": "The agent task description",
    "properties": {
        "agent_description": {
            "description": "The agent task description",
            "title": "agent_description",
            "type": "string"
        },
    },
    "required": [
        "agent_description",
    ],
    "title": "agent_description",
    "type": "object"
}

action_decision_flow_schema = {
    "description": "Next action decision",
    "properties": {
        "decision": {
            "description": "Answer Yes in case the decision is to rewrite the function, otherwise answer No",
            "title": "decision",
            "type": "boolean"
        },
    },
    "required": [
        "decision",
    ],
    "title": "decision",
    "type": "object"
}

action_decision_agent_schema = {
    "description": "Next action decision",
    "properties": {
        "decision": {
            "description": "Answer Yes in case the decision is to breakdown the task, otherwise answer No",
            "title": "decision",
            "type": "boolean"
        },
        "reason": {
            "description": "The reason for the decision",
            "title": "reason",
            "type": "string"}
    },
    "required": [
        "decision", "reason"
    ],
    "title": "decision",
    "type": "object"
}

updating_flow_schema = {
    "description": "The updated flow",
    "properties": {
        "code": {
            "description": "The updated code",
            "title": "code",
            "type": "string"
        },
    },
    "required": [
        "code",
    ],
    "title": "updated_flow",
    "type": "object"
}

metric_generator_schema = {
    "$defs": {
        "MetricScale": {
            "description": "Prompts for each evaluation scale for a given evaluation metric.",
            "properties": {
                "deficient_desc": {
                    "description": "1. Deficient Performance on (a few words about the metric go here) (1): a single sentence description of what should be checked in the task output such that it classifies as a Deficient level of performance relative to this metric.",
                    "title": "Deficient_Description",
                    "type": "string"
                },
                "adequate_desc": {
                    "description": "2. Adequate Performance on (a few words about the metric go here) (2): a single sentence description of what should be checked in the task output such that it classifies as a Adequate level of performance relative to this metric.",
                    "title": "Adequate_Description",
                    "type": "string"
                },
                "competent_desc": {
                    "description": "3. Competent Performance on (a few words about the metric go here) (3): a single sentence description of what should be checked in the task output such that it classifies as a Competent level of performance relative to this metric.",
                    "title": "Competent_Description",
                    "type": "string"
                },
                "proficient_desc": {
                    "description": "4. Proficient Performance on (a few words about the metric go here) (4): a single sentence description of what should be checked in the task output such that it classifies as a Proficient level of performance relative to this metric.",
                    "title": "Proficient_Description",
                    "type": "string"
                },
                "exemplary_desc": {
                    "description": "5. Exemplary Performance on (a few words about the metric go here) (5): a single sentence description of what should be checked in the task output such that it classifies as a Exemplary level of performance relative to this metric.",
                    "title": "Exemplary_Description",
                    "type": "string"
                }
            },
            "required": [
                "deficient_desc",
                "adequate_desc",
                "competent_desc",
                "proficient_desc",
                "exemplary_desc"
            ],
            "title": "Metric_Scale_Object",
            "type": "object"
        },
        "Metric": {
            "description": "Details about a particular Evaluation Metric",
            "properties": {
                "metric_name": {
                    "description": "The name of the evaluation metric, in a few words, that will serve as the area of diagnostic evaluation for the task.",
                    "title": "Metric_Name",
                    "type": "string"
                },
                "metric_desc": {
                    "description": "Explanation, in not more than 3 sentences, what the above metric means and what it's trying to assess.",
                    "title": "Metric_Description",
                    "type": "string"
                },
                "metric_prompt": {
                    "description": "A prompt that will be used as input to an external evaluator to evaluate the assistant's performance on this metric.",
                    "title": "Metric_Prompt",
                    "type": "string"
                },
                "metric_category": {
                    "description": "Determine which part of the agent flow this metric is assessing (e.g., RAG, Tools, input, output, etc.)",
                    "title": "Metric_Category",
                    "type": "string"
                },
                "is_metric_end2end": {
                    "description": "Determine if the metric is testing the end-to-end performance of the agent or is it testing a sub-component of the agent.",
                    "title": "Is_Metric_End2End",
                    "type": "boolean"
                },

                "metric_scale": {
                    "description": "A list of descriptions for each scale of assessment from 1 to 5 for the given metric. Note that in the above metric prompt structure, 1 represents the lowest level of performance and 5 represents the best level of performance.",
                    "items": {
                        "$ref": "#/$defs/MetricScale"
                    },
                    "title": "Metric_Scale",
                    "type": "array"
                }
            },
            "required": ["metric_name", "metric_desc", "metric_prompt", "metric_scale"],
            "title": "Metric",
            "type": "object"
        }
    },
    "description": "The collection of Metrics.",
    "properties": {
        "metrics_list": {
            "description": "The list of all possible metrics that would be important to assess whether this assistant performed the given task perfectly or not.",
            "items": {
                "$ref": "#/$defs/Metric"
            },
            "title": "Metrics_List",
            "type": "array"
        }
    },
    "required": [
        "metrics_list"
    ],
    "title": "Metrics_Object",
    "type": "object"
}
metric_merge_schema = metric_generator_schema

def update_classification_prediction_schema(label_schema: list) -> dict:
    """
  Updates the classification prediction schema with the label schema from the yaml file
  :param yaml_data: The yaml data
  """

    classification_prediction_schema['$defs']['Result']['properties']['prediction']['enum'] = label_schema
    classification_prediction_schema['$defs']['Result']['properties']['prediction'][
        'description'] += 'The answer must be one of the following options: {} !!'.format(label_schema)
    return classification_prediction_schema

def step_prompt_parser(response: dict) -> dict:
    """
    Parse the response of the step prompt
    :param response: The response
    :return: The parsed response
    """
    tools_metadata = {t['tool_name']: t['description'] for t in response['tools_description']}
    if len(tools_metadata) == 0:
        return {'prompt': response['prompt'], 'tools_description': "The agent doesn't have any available tool!!",
                'tools_metadata': tools_metadata}
    description_str = dict_to_prompt_text(tools_metadata)
    return {'prompt': response['prompt'], 'tools_description': description_str,  'tools_metadata': tools_metadata}
