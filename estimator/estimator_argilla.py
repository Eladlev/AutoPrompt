import argilla as rg
import time
import pandas as pd
from argilla.client.singleton import active_client
from utils.config import Color
from dataset.base_dataset import DatasetBase
import json
import webbrowser
import base64

class ArgillaEstimator:
    """
    The ArgillaEstimator class is responsible to generate the GT for the dataset by using Argilla interface.
    In particular using the text classification mode.
    """
    def __init__(self, opt):
        """
        Initialize a new instance of the ArgillaEstimator class.
        """
        try:
            self.opt = opt
            rg.init(
                api_url=opt.api_url,
                api_key=opt.api_key,
                workspace=opt.workspace
            )
            self.time_interval = opt.time_interval
        except:
            raise Exception("Failed to connect to argilla, check connection details")

    @staticmethod
    def initialize_dataset(dataset_name: str, label_schema: set[str]):
        """
        Initialize a new dataset in the Argilla system
        :param dataset_name: The name of the dataset
        :param label_schema: The list of classes
        """
        try:
            settings = rg.TextClassificationSettings(label_schema=label_schema)
            rg.configure_dataset_settings(name=dataset_name, settings=settings)
        except:
            raise Exception("Failed to create dataset")

    @staticmethod
    def upload_missing_records(dataset_name: str, batch_id: int, batch_records: pd.DataFrame):
        """
        Update the Argilla dataset by adding missing records from batch_id that appears in batch_records
        :param dataset_name: The dataset name
        :param batch_id: The batch id
        :param batch_records: A dataframe of the batch records
        """
        #TODO: sort visualization according to batch_id descending
        query = "metadata.batch_id:{}".format(batch_id)
        result = rg.load(name=dataset_name, query=query)
        df = result.to_pandas()
        if len(df) == len(batch_records):
            return
        if df.empty:
            upload_df = batch_records
        else:
            merged_df = pd.merge(batch_records, df['text'], on='text', how='left', indicator=True)
            upload_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        record_list = []
        for index, row in upload_df.iterrows():
            config = {'text': row['text'], 'metadata': {"batch_id": row['batch_id'], 'id': row['id']}, "id": row['id']}
            # if not (row[['prediction']].isnull().any()):
            #     config['prediction'] = row['prediction']  # TODO: fix it incorrect type!!!
            if not(row[['annotation']].isnull().any()):  # TODO: fix it incorrect type!!!
                config['annotation'] = row['annotation']
            record_list.append(rg.TextClassificationRecord(**config))
        rg.log(records=record_list, name=dataset_name)

    def calc_usage(self):
        """
        Dummy function to calculate the usage of the estimator
        """
        return 0

    def apply(self, dataset: DatasetBase, batch_id: int):
        """
        Apply the estimator on the dataset. The function enter to infinite loop until all the records are annotated.
        Then it update the dataset with all the annotations
        :param dataset: DatasetBase object, contains all the processed records
        :param batch_id: The batch id to annotate
        """
        current_api = active_client()
        try:
            rg_dataset = current_api.datasets.find_by_name(dataset.name)
        except:
            self.initialize_dataset(dataset.name, dataset.label_schema)
            rg_dataset = current_api.datasets.find_by_name(dataset.name)
        batch_records = dataset[batch_id]
        if batch_records.empty:
            return []
        self.upload_missing_records(dataset.name, batch_id, batch_records)
        data = {'metadata': {'batch_id': [str(batch_id)]}}
        json_data = json.dumps(data)
        encoded_bytes = base64.b64encode(json_data.encode('utf-8'))
        encoded_string = str(encoded_bytes, "utf-8")
        url_link = self.opt.api_url + '/datasets/' + self.opt.workspace + '/' \
                   + dataset.name + '?query=' + encoded_string
        print(f"{Color.GREEN}Waiting for annotations from batch {batch_id}:\n{url_link}{Color.END}")
        webbrowser.open(url_link)
        while True:
            query = "(status:Validated OR status:Discarded) AND metadata.batch_id:{}".format(batch_id)
            search_results = current_api.search.search_records(
                name=dataset.name,
                task=rg_dataset.task,
                size=0,
                query_text=query,
            )
            if search_results.total == len(batch_records):
                result = rg.load(name=dataset.name, query=query)
                df = result.to_pandas()[['text', 'annotation', 'metadata', 'status']]
                df["annotation"] = df.apply(lambda x: 'Discarded' if x['status']=='Discarded' else x['annotation'], axis=1)
                df = df.drop(columns=['status'])
                df['id'] = df.apply(lambda x: x['metadata']['id'], axis=1)
                return df
            time.sleep(self.time_interval)
