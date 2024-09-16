from utils.llm_chain import ChainWrapper, get_chain_metadata
from pathlib import Path
from dataset.base_dataset import DatasetBase
import pandas as pd
from utils.config import get_t2i_model
import yaml
import asyncio

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

        self.image_generator = get_t2i_model(opt.t2i)

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

    def apply_dataframe(self, records: pd.DataFrame):
        """
        Apply the estimator on a dataframe
        :param records: The record
        """
        # num_images = len(records)
        # prompt = self.cur_instruct['prompt']
        img_urls = self.image_generator(prompt=self.cur_instruct['prompt'],
                                        num_images=len(records))
        # prompts = [self.cur_instruct['prompt']]*num_images
        # img_urls = asyncio.run(self.image_generator(prompts))
        # img_urls = [im.url for response in responses for im in response.data]
        records[self.mode] = img_urls
        return records

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
