from estimator.estimator_llm import LLMEstimator
import json


def set_function_from_iterrow(func):
    def wrapper(dataset):
        dataset['score'] = dataset.apply(func, axis=1)
        return dataset

    return wrapper


def set_ranking_function(params):
    evaluator = LLMEstimator(params)
    evaluator.init_chain(params.label_schema)
    evaluator.mode = 'score'
    def wrapper(dataset):
        generation_dataset = dataset.copy()
        generation_dataset['text'] = '###User input:\n' + generation_dataset['text'] + '\n####model prediction:\n' + generation_dataset['prediction']

        generation_dataset = evaluator.apply_dataframe(generation_dataset)

        # Convert the list of dictionaries having metric_name, metric_score, metric_reasoning to JSON string
        generation_dataset.score = generation_dataset.score.apply(lambda x: json.dumps(x))
        
        dataset.score = generation_dataset.score
        return dataset
    return wrapper
