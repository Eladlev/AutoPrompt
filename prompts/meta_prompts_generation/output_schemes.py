# A file containing the json schema for the output of all the LLM chains
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

initial_schema = step_samples_schema = {
  "description": "A List of all results",
  "properties": {
    "samples": {
      "description": "Each sample is a string containing only the prompt sample content, without any additional information",
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


class MetricScale(BaseModel):
  deficient_desc: str = Field(
    description="1. Deficient Performance on (a few words about the metric go here) (1): a single sentence description of what should be checked in the task output such that it classifies as a Deficient level of performance relative to this metric.")
  adequate_desc: str = Field(
    description="2. Adequate Performance on (a few words about the metric go here) (2): a single sentence description of what should be checked in the task output such that it classifies as a Adequate level of performance relative to this metric.")
  competent_desc: str = Field(
    description="3. Competent Performance on (a few words about the metric go here) (3): a single sentence description of what should be checked in the task output such that it classifies as a Competent level of performance relative to this metric.")
  proficient_desc: str = Field(
    description="4. Proficient Performance on (a few words about the metric go here) (4): a single sentence description of what should be checked in the task output such that it classifies as a Proficient level of performance relative to this metric.")
  exemplary_desc: str = Field(
    description="5. Exemplary Performance on (a few words about the metric go here) (5): a single sentence description of what should be checked in the task output such that it classifies as a Exemplary level of performance relative to this metric.")


class Metric(BaseModel):
  metric_name: str = Field(
    description="The name of the evaluation metric, in a few words, that will serve as the area of diagnostic evaluation for the task.")
  metric_desc: str = Field(
    description="Explanation, in not more than 3 sentences, what the above metric means and what it's trying to assess.")
  metric_prompt: str = Field(
    description="A prompt that will be used as input to an external evaluator agent to evaluate the assistant's performance based on the metric name and metric description generated above. This prompt should start with the phrase: 'Evaluate the performance of our agent using a five-point assessment scale that emphasizes', followed by a few words about the metric name and description, followed by another sentence starting with the phrase: 'Assign a single score of either 1, 2, 3, 4, or 5, with each level representing different degrees of perfection with respect to the', followed by just a few words again summarizing the metric name and description.")
  metric_scale: List[MetricScale] = Field(
    description="A list of descriptions for each scale of assessment from 1 to 5 for the given metric. Note that in the above metric prompt structure, 1 represents the lowest level of performance and 5 represents the best level of performance.")


class MetricGeneratorFlow(BaseModel):
  metrics_list: List[Metric] = Field(
    description="The list of all possible rules that would be important to assess whether this assistant performed the above task perfectly or not.")

metric_generator_schema = MetricGeneratorFlow

def update_classification_prediction_schema(label_schema:list)->dict:
  """
  Updates the classification prediction schema with the label schema from the yaml file
  :param yaml_data: The yaml data
  """

  classification_prediction_schema['$defs']['Result']['properties']['prediction']['enum'] = label_schema
  classification_prediction_schema['$defs']['Result']['properties']['prediction'][
    'description'] += 'The answer must be one of the following options: {} !!'.format(label_schema)
  return classification_prediction_schema