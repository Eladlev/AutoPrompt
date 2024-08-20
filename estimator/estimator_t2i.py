from utils.llm_chain import ChainWrapper, get_chain_metadata
from pathlib import Path
from dataset.base_dataset import DatasetBase
import pandas as pd
from openai import OpenAI
import yaml

LLM_ENV = yaml.safe_load(open('config/llm_env.yml', 'r'))

class T2IEstimator:
    """
    A wrapper for an estimator using LLM
    """

    def __init__(self, opt):
        """
        Initialize a new instance of the T2IEstimator class.
        :param opt: The configuration file (EasyDict)
        """
        self.opt = opt
        self.chain = None
        self.mini_batch_size = opt.mini_batch_size
        self.mode = opt.mode
        self.num_workers = opt.num_workers
        if 'instruction' in opt.keys():
            self.cur_instruct = opt.instruction
        else:
            self.cur_instruct = None

        self.client = OpenAI(api_key=opt.get('api_key', LLM_ENV['openai']['OPENAI_API_KEY']))

    @staticmethod
    def generate_sample_image(sample_id: int, url: str) -> str:
        """
        Generate a sample image for the chain prompt
        :param sample_id: The sample id
        :param url: The url of the sample
        :return: A string of the sample ID and sample url for the prompt
        """
        return f"ID: {sample_id};  Sample: {url}\n"

    def describe_image_batches(self, image_url):
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        image_desc_str = response.choices[0].message.content
        return image_desc_str

    def generate_image_batches(self, prompt):

        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            n=1,
        )

        image_url = response.data[0].url
        return image_url

    def calc_usage(self) -> float:
        """"
        Calculate the usage of the estimator
        """
        if self.chain is not None:
            return self.chain.accumulate_usage
        else:
            return 0

    def init_chain(self, label_schema: set[str]):
        """
        Initialize the chain
        :param label_schema: The label schema
        """
        chain_metadata = get_chain_metadata(Path(self.opt.prompt), retrieve_module=True)
        if hasattr(chain_metadata['module'], 'update_classification_prediction_schema'):
            chain_metadata['json_schema'] = chain_metadata['module'].update_classification_prediction_schema(
                chain_metadata['json_schema'],
                label_schema
            )
        self.chain = None
        # self.chain = ChainWrapper(self.opt.llm,
        #                           self.opt.prompt,
        #                           chain_metadata['json_schema'],
        #                           chain_metadata['parser_func'])

    def apply_dataframe(self, record: pd.DataFrame):
        """
        Apply the estimator on a dataframe
        :param record: The record
        """
        chain_input = ''
        mini_batch_inputs = []
        record[self.mode] = 'Discarded'
        # prepare all the inputs for the chains
        for i, row in record.iterrows():
            chain_input += self.generate_sample_image(i, row['text'])
            if ((i + 1) % self.mini_batch_size) == 0:
                mini_batch_inputs.append({'batch_size': self.mini_batch_size,
                                          'task_instruction': self.cur_instruct,
                                          'samples': chain_input})
                chain_input = ''
        if not (chain_input == ''):
            mini_batch_inputs.append({'batch_size': self.mini_batch_size,
                                      'task_instruction': self.cur_instruct,
                                      'samples': chain_input})

        if self.chain is not None:
            all_results = self.chain.batch_invoke(mini_batch_inputs, self.num_workers)
            union_results = [element for sublist in all_results for element in sublist['results']]
        else:
            all_results = []
            for ib, mini_batch_input in enumerate(mini_batch_inputs):
                response = self.client.images.generate(
                    model=self.opt.llm.name,
                    # prompt=mini_batch_input["task_instruction"],
                    prompt=mini_batch_input["samples"].split("Sample: ")[1].strip(),
                    **self.opt.llm.model_kwargs)
                image_url = response.data[0].url
                sample_id = int(mini_batch_input["samples"].split(";")[0].replace("ID: ", ""))
                all_results.append({"id": sample_id, "prediction": image_url})
            union_results = all_results

        for res in union_results:
            record.loc[res['id'], self.mode] = res['prediction']
        return record

    def apply(self, dataset: DatasetBase, idx: int, leq: bool = False):
        """
        Apply the estimator on the batches up to idx (includes), it then updates the annotation field
        if self.mode is 'annotation', otherwise it update the prediction field.
        :param dataset: The dataset
        :param idx: The current batch index
        :param leq: If True, apply on all the batches up to idx (includes), otherwise apply only on idx
        """
        if self.chain is None:
            self.init_chain(dataset.label_schema)
        if leq:
            batch_records = dataset.get_leq(idx)
        else:
            batch_records = dataset[idx]
        return self.apply_dataframe(batch_records)
