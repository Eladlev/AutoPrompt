import os.path
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import csv

from utils.dedup import Dedup

class DatasetBase:
    """
    This class store and manage all the dataset records (including the annotations and prediction)
    """

    def __init__(self, config):
        if config.records_path is None:
            self.records = pd.DataFrame(columns=['id', 'text', 'prediction',
                                                 'annotation', 'metadata', 'score', 'batch_id'])
        else:
            self.records = pd.read_csv(config.records_path)
        dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        self.name = config.name + '__' + dt_string
        self.label_schema = config.label_schema
        self.dedup = Dedup(config)
        self.sample_size = config.get("sample_size", 3)
        self.semantic_sampling = config.get("semantic_sampling", False)
        if not config.get('dedup_new_samples', False):
            self.remove_duplicates = self._null_remove

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.records)

    def __getitem__(self, batch_idx):
        """
        Return the batch idx.
        """
        extract_records = self.records[self.records['batch_id'] == batch_idx]
        extract_records = extract_records.reset_index(drop=True)
        return extract_records

    def get_leq(self, batch_idx):
        """
        Return all the records up to batch_idx (includes).
        """
        extract_records = self.records[self.records['batch_id'] <= batch_idx]
        extract_records = extract_records.reset_index(drop=True)
        return extract_records

    def add(self, sample_list: dict = None, batch_id: int = None, records: pd.DataFrame = None):
        """
        Add records to the dataset.
        :param sample_list: The samples to add in a dict structure (only used in case record=None)
        :param batch_id: The batch_id for the upload records (only used in case record= None)
        :param records: dataframes, update using pandas
        """
        if records is None:
            records = pd.DataFrame([{'id': len(self.records) + i, 'text': sample, 'batch_id': batch_id} for
                       i, sample in enumerate(sample_list)])
        self.records = pd.concat([self.records, records], ignore_index=True)

    def update(self, records: pd.DataFrame):
        """
        Update records in dataset.
        """
        # Ignore if records is empty
        if len(records) == 0:
            return

        # Set 'id' as the index for both DataFrames
        records.set_index('id', inplace=True)
        self.records.set_index('id', inplace=True)

        # Update using 'id' as the key
        self.records.update(records)

        # Remove null annotations
        if len(self.records.loc[self.records["annotation"]=="Discarded"]) > 0:
            discarded_annotation_records = self.records.loc[self.records["annotation"]=="Discarded"]
            #TODO: direct `discarded_annotation_records` to another dataset to be used later for corner-cases
            self.records = self.records.loc[self.records["annotation"]!="Discarded"]

        # Reset index
        self.records.reset_index(inplace=True)

    def modify(self, index: int, record: dict):
        """
        Modify a record in the dataset.
        """
        self.records[index] = record

    def apply(self, function, column_name: str):
        """
        Apply function on each record.
        """
        self.records[column_name] = self.records.apply(function, axis=1)

    def save_dataset(self, path: Path):
        self.records.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def load_dataset(self, path: Path):
        """
        Loading dataset
        :param path: path for the csv
        """
        if os.path.isfile(path):
            self.records = pd.read_csv(path, dtype={'annotation': str, 'prediction': str, 'batch_id': int})
        else:
            logging.warning('Dataset dump not found, initializing from zero')

    def remove_duplicates(self, samples: list) -> list:
        """
        Remove (soft) duplicates from the given samples
        :param samples: The samples
        :return: The samples without duplicates
        """
        dd = self.dedup.copy()
        df = pd.DataFrame(samples, columns=['text'])
        df_dedup = dd.sample(df, operation_function=min)
        return df_dedup['text'].tolist()

    def _null_remove(self, samples: list) -> list:
        # Identity function that returns the input unmodified
        return samples

    def sample_records(self, n: int = None) -> pd.DataFrame:
        """
        Return a sample of the records after semantic clustering
        :param n: The number of samples to return
        :return: A sample of the records
        """
        n = n or self.sample_size
        if self.semantic_sampling:
            dd = self.dedup.copy()
            df_samples = dd.sample(self.records).head(n)

            if len(df_samples) < n:
                df_samples = self.records.head(n)
        else:
            df_samples = self.records.sample(n)
        return df_samples

    @staticmethod
    def samples_to_text(records: pd.DataFrame) -> str:
        """
        Return a string that organize the samples for a meta-prompt
        :param records: The samples for the step
        :return: A string that contains the organized samples
        """
        txt_res = '##\n'
        for i, row in records.iterrows():
            txt_res += f"Sample:\n {row.text}\n#\n"
        return txt_res


