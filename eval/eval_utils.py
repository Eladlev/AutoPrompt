from estimator.estimator_llm import LLMEstimator


def set_function_from_iterrow(func):
    def wrapper(dataset):
        dataset['score'] = dataset.apply(func, axis=1)
        return dataset

    return wrapper


def set_ranking_function(params):
    evaluator = LLMEstimator(params)
    evaluator.init_chain(params.label_schema)
    evaluator.cur_instruct = params.instruction
    evaluator.mode = 'score'
    def wrapper(dataset):
        dataset = evaluator.apply_dataframe(dataset)
        dataset.score = dataset.score.astype(int)
        return dataset
    return wrapper
