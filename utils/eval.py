import pandas as pd
import numpy as np
from dataset.base_dataset import DatasetBase


class Eval:
    """
    The Eval class is responsible to calculate the score and the large errors
    """

    def __init__(self, config):
        """
        Initialize a new instance of the Eval class.
        :param config: The configuration file (EasyDict)
        """
        self.score_func = config.score_function
        self.num_boundary_predictions = config.num_boundary_predictions
        self.th = config.th
        self.dataset = None
        self.mean_score = None
        self.errors = None
        self.history = []

    def eval_score(self, dataset: DatasetBase) -> float:
        """
        Calculate the score on each row and return the mean score.
        :param dataset: DatasetBase, the pipeline dataset
        :return: The mean score
        """
        dataset.apply(self.score_func, column_name='score')
        self.mean_score = round(dataset.records['score'].mean(), 2)
        return self.mean_score

    def get_min_score(self):
        """
        Return the minimum 'mean score' (with respect to all history epochs) and the epoch index of the minimum score
        :return: The epoch index of the minimum score, and the minimum score
        """
        min_idx = np.argmax([epoch['score'] for epoch in self.history])
        return min_idx, self.history[min_idx]['score']

    @staticmethod
    def large_error_to_str(error_df: pd.DataFrame, num_large_errors_per_label: int) -> str:
        """
        Return a string that contains the large errors
        :param error_df: A dataframe contains all the mislabeled samples
        :param num_large_errors_per_label: The (maximum) number of large errors per label
        :return: A string that contains the large errors that is used in the meta-prompt
        """
        error_res_df = pd.DataFrame(columns=['annotation', 'text', 'score', 'prediction'])
        label_schema = error_df['annotation'].unique()

        for label in label_schema:
            cur_df = error_df[error_df['annotation'] == label][:num_large_errors_per_label]
            # need to cast the boolean explicitly to avoid future warning:
            # .../AutoPrompt/utils/eval.py:55: FutureWarning: In a future version,
            # object-dtype columns with all-bool values will not be included in reductions with bool_only=True.
            # Explicitly cast to bool dtype instead.
            # df["var1"] = df["var1"].astype(bool)
            if len(cur_df) > 0:
                error_res_df = pd.concat([error_res_df, cur_df[error_res_df.columns]], ignore_index=True)
        txt_res = '##Failure Cases:\n'
        error_res_df = error_res_df.sample(frac=1.0, random_state=42)
        for i, row in error_res_df.iterrows():
            txt_res += f"Sample: {row.text}\nPrediction: {row.prediction}, GT: {row.annotation}\n#\n"
        return txt_res

    @staticmethod
    def sample_to_text(sample: dict, num_errors_per_label: int = 0, is_score: bool = True) -> str:
        """
        Return a string that organize the information of from the step run for the meta-prompt
        :param sample: The eval information for specific step
        :param num_errors_per_label: The max number of large errors per class that will appear in the meta-prompt
        :param is_score: If True, add the score information to the meta-prompt
        :return: A string that contains the information of the step run
        """
        if is_score:
            return f"####\n##Prompt:\n{sample['prompt']}\n##Score: {sample['score']}\n{Eval.large_error_to_str(sample['errors'], num_errors_per_label)}####\n"
        else:
            return f"####\n##Prompt:\n{sample['prompt']}\n{Eval.large_error_to_str(sample['errors'], num_errors_per_label)}####\n "

    def add_history(self, prompt: str):
        """
        Add the current step information to the history
        :param prompt: The current prompt
        """
        self.history.append({'prompt': prompt, 'score': self.mean_score, 'errors': self.errors})

    def extract_errors(self) -> pd.DataFrame:
        """
        Extract the errors from the dataset
        :return: records that contains the errors
        """
        df = self.dataset
        err_df = df[df['score'] < self.th]
        err_df.sort_values(by=['score'])
        self.errors = err_df
        return self.errors

    def extract_correct(self) -> pd.DataFrame:
        """
        Extract the correct samples from the dataset
        :return: records that contains the correct samples
        """
        df = self.dataset
        return df[df['score'] > self.th]

    def extract_boundary_predictions(self) -> pd.DataFrame:
        """
        Extract boundary samples on which the model is uncertain
        :return: records that contains boundary samples
        """
        pass
