from utils.llm_chain import ChainWrapper
from utils.config import load_yaml
import pickle
import pandas as pd

config_params = load_yaml('config/config_default.yml')

chain_analysis = ChainWrapper(config_params['llm'], 'prompts/meta_prompts_generation/error_analysis.prompt',
                     None,
                     None)
# This is the dataset with all the predictions
dataset = pd.read_csv('./dump/dataset.csv')

# This is all the additional information that is needed for the analysis (e.g., metrics and task description)
info = pickle.load(open('./dump/analysis.pkl', 'rb'))

prompt = info["prompt"] 
score_info = info['score_info']
task_description = info["task_description"]
accuracy = info['accuracy']


def extract_relevant_inputs(dataset, task_description, prompt, score_info, accuracy):
    """
    Extracts relevant inputs for a given prompt from the dataset based on the task description, score information, and accuracy.

    This function iterates over the dataset and selects entries that match the provided task description, prompt, score information, and accuracy. The selected entries are considered relevant for the given prompt.

    :param dataset: The dataset to extract inputs from. This should be a list of dictionaries.
    :param task_description: The description of the task to match in the dataset entries.
    :param prompt: The prompt to match in the dataset entries.
    :param score_info: The score information to match in the dataset entries.
    :param accuracy: The accuracy to match in the dataset entries.
    :return: A list of relevant inputs.
    """

    df = dataset
    num_large_errors_per_label = 10
    error_threshold = df['score'].mean()

    error_df = df[df['score'] <= error_threshold]
    error_df = error_df.sort_values(by='score', ascending=True)
    error_df = error_df[:num_large_errors_per_label]

    # Retrieve all columns that start with 'score_'
    metrics = [col[6:] for col in error_df.columns if col.startswith('score_')]

    failure_cases = ''
    for index, row in error_df.iterrows():
        metric_result = ''
        for metric in metrics:
            if row['score_{}'.format(metric)] <= error_threshold:
                metric_result+= f"#{metric}: {row['score_{}'.format(metric)]}\n{metric} score reason: {row['reasoning_{}'.format(metric)]}\n"
        failure_cases += f"###Sample text\n{row['text']}\n###Agent response issues:\n{metric_result}\n"

    return chain_analysis.invoke({
        'task_description': task_description,
        'prompt': prompt,
        'score_info': score_info,
        'accuracy': accuracy,
        'failure_cases': failure_cases
    })


extract_relevant_inputs(dataset, task_description, prompt, score_info, accuracy)