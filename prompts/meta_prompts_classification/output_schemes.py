# A file containing the json schema for the output of all the LLM chains

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

def update_classification_prediction_schema(label_schema:list)->dict:
  """
  Updates the classification prediction schema with the label schema from the yaml file
  :param yaml_data: The yaml data
  """

  classification_prediction_schema['$defs']['Result']['properties']['prediction']['enum'] = label_schema
  classification_prediction_schema['$defs']['Result']['properties']['prediction'][
    'description'] += 'The answer must be one of the following options: {} !!'.format(label_schema)
  return classification_prediction_schema