import os.path
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime



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

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.records)

    def __getitem__(self, batch_idx):
        """
        Return the batch idx.
        """
        return self.records[self.records['batch_id'] == batch_idx]

    def get_leq(self, batch_idx):
        """
        Return all the records up to batch_idx (includes).
        """
        return self.records[self.records['batch_id'] <= batch_idx]

    def add(self, sample_list: dict = None, batch_id: int = None, records: pd.DataFrame = None):
        """
        Add records to the dataset.
        :param sample_list: The samples to add in a dict structure (only used in case record=None)
        :param batch_id: The batch_id for the upload records (only used in case record= None)
        :param records: dataframes, update using pandas
        """
        if records is None:
            records = [{'id': len(self.records) + i, 'text': sample, 'batch_id': batch_id} for
                       i, sample in enumerate(sample_list)]
        self.records = self.records.append(records, ignore_index=True)

    def update(self, records: pd.DataFrame):
        """
        Update records in dataset.
        """
        # Set 'id' as the index for both DataFrames
        records.set_index('id', inplace=True)
        self.records.set_index('id', inplace=True)

        # Update using 'id' as the key
        self.records.update(records)
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
        self.records.to_csv(path)

    def load_dataset(self, path: Path):
        """
        Loading dataset
        :param path: path for the csv
        """
        if os.path.isfile(path):
            self.records = pd.read_csv(path)
        else:
            logging.warning('Dataset dump not found, initializing from zero')
