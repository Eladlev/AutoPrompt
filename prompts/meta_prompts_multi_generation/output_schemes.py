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
  },
  "required": [
    "prompt",
  ],
  "title": "Suggested_Prompt",
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
            "description": "A prompt that will be used as input to an external evaluator agent to evaluate the assistant's performance based on the metric name and metric description generated above. This prompt should start with the phrase: 'Evaluate the performance of our agent using a five-point assessment scale that emphasizes', followed by a few words about the metric name and description, followed by another sentence starting with the phrase: 'Assign a single score of either 1, 2, 3, 4, or 5, with each level representing different degrees of perfection with respect to the', followed by just a few words again summarizing the metric name and description.",
            "title": "Metric_Prompt",
            "type": "string"
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

def update_classification_prediction_schema(label_schema:list)->dict:
  """
  Updates the classification prediction schema with the label schema from the yaml file
  :param yaml_data: The yaml data
  """

  classification_prediction_schema['$defs']['Result']['properties']['prediction']['enum'] = label_schema
  classification_prediction_schema['$defs']['Result']['properties']['prediction'][
    'description'] += 'The answer must be one of the following options: {} !!'.format(label_schema)
  return classification_prediction_schema
