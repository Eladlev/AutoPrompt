# A file containing the json schema for the output of all the LLM chains
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Deque, List, Optional, Tuple
from agent.agent_instantiation import Variable



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


class MetricResults(BaseModel):
    # The metric results
    metric_name: str = Field(description="The metric name")
    metric_prompt: str = Field(description="The metric prompt")

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

step_prompt_schema = {
    "description": "A prompt suggestion which expect to get high score, and the associated score prediction",
    "properties": {
        "prompt": {
            "description": "The prompt prediction",
            "title": "Prompt",
            "type": "string"
        },
        "score": {
            "description": "The score prediction",
            "title": "Score",
            "type": "number"
        }
    },
    "required": [
        "prompt",
        "score"
    ],
    "title": "Suggested_Prompt",
    "type": "object"
}


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
    },
    "required": [
        "decision",
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

def update_classification_prediction_schema(label_schema: list) -> dict:
    """
  Updates the classification prediction schema with the label schema from the yaml file
  :param yaml_data: The yaml data
  """

    classification_prediction_schema['$defs']['Result']['properties']['prediction']['enum'] = label_schema
    classification_prediction_schema['$defs']['Result']['properties']['prediction'][
        'description'] += 'The answer must be one of the following options: {} !!'.format(label_schema)
    return classification_prediction_schema

