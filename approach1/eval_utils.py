import json
from estimator.estimator_llm import LLMEstimator

def set_function_from_iterrow(func):
    def wrapper(dataset):
        dataset['score'] = dataset.apply(func, axis=1)
        return dataset

    return wrapper


def calculate_scores(text, prediction):
    prompt = load_prompt_from_file('prompts/score_evaluator.prompt')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",      #change to gpt-4
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(text=text, prediction=prediction)}
        ],
        max_tokens=500,
        temperature=0
    )

    try:
        scores = json.loads(response['choices'][0]['message']['content'].strip())
    except json.JSONDecodeError:
        raise ValueError('The response from the API could not be parsed as JSON')

    return scores

def load_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def set_ranking_function(params):
    evaluator = LLMEstimator(params)
    evaluator.init_chain(params.label_schema)
    evaluator.mode = 'score'

    def wrapper(dataset):
        generation_dataset = dataset.copy()
        generation_dataset['text'] = '###User input:\n' + generation_dataset['text'] + '\n####model prediction:\n' + generation_dataset['prediction']
        
        # List to store JSON strings of multiple scores and reasoning
        score_list = []
        for index, row in generation_dataset.iterrows():
            scores_json = calculate_scores(row['text'], row['prediction'])
            score_list.append(json.dumps(scores_json))  # Convert to JSON string
        
        # Store the JSON list in the score column
        generation_dataset['score'] = score_list
        dataset['score'] = generation_dataset['score']
        
        return dataset
    return wrapper


