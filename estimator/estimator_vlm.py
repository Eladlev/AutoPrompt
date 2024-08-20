from utils.llm_chain import ChainWrapper, get_chain_metadata
from pathlib import Path
from dataset.base_dataset import DatasetBase
import pandas as pd
from openai import OpenAI
import numpy as np
from scipy.spatial import distance
import yaml

LLM_ENV = yaml.safe_load(open('config/llm_env.yml', 'r'))


class VLMEstimator:
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
        # self.mini_batch_size = opt.mini_batch_size
        # self.mode = opt.mode
        # self.num_workers = opt.num_workers
        self.image_description_model_name = opt.get("image_description_model_name", "gpt-4-vision-preview")
        self.text_embedding_model_name = opt.get("text_embedding_model_name", "text-embedding-3-small")
        self.max_tokens = opt.get("max_tokens", 512)
        self.cur_instruct = opt.get("instruction", None)
        # if 'instruction' in opt.keys():
        #     self.cur_instruct = opt.instruction
        # else:
        #     self.cur_instruct = None

        self.client = OpenAI(api_key=opt.get('api_key', LLM_ENV['openai']['OPENAI_API_KEY']))

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
        self.chain = ChainWrapper(self.opt.llm, self.opt.prompt, chain_metadata['json_schema'],
                                  chain_metadata['parser_func'])

    def describe_image(self, image_url):
        image_description_prompt = """
        Describe the contents of this image. 
        Pay close attention to the object of the image and be as descriptive as possible.
        Describe the background, the surroundings, the atmosphere, the lighting and make sure to note the style of the image. 
        Describe the foreground object or main focus of the image in as much details as possible. 
        """
        response = self.client.chat.completions.create(
            model=self.image_description_model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": image_description_prompt
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
            max_tokens=self.max_tokens,
        )

        image_desc_str = response.choices[0].message.content
        return image_desc_str

    def get_embedding(self, text_input):
        text_input = text_input.replace("\n", " ")
        response = self.client.embeddings.create(input=[text_input], model=self.text_embedding_model_name)
        text_embedding = response.data[0].embedding
        return np.array(text_embedding, dtype=np.float32)

    def get_similarity_score(self, image_desc, prompt_input):
        image_desc_embedding = self.get_embedding(image_desc)
        prompt_input_embedding = self.get_embedding(prompt_input)
        embedding_dist = distance.cosine(image_desc_embedding, prompt_input_embedding)
        similarity_score = embedding_dist
        return similarity_score

    def apply_dataframe(self, records: pd.DataFrame):
        """
        Apply the estimator on a dataframe
        :param records: The record
        """
        # chain_input = ''
        # mini_batch_inputs = []
        # records[self.mode] = 'Discarded'
        # prepare all the inputs for the chains
        for i, row in records.iterrows():
            image_url = row["prediction"]
            image_description = self.describe_image(image_url)
            records.at[i, "annotation"] = image_description
            prompt_input = row["text"]
            similarity_score = self.get_similarity_score(image_description, prompt_input)
            records.at[i, "score"] = similarity_score

        #     if ((i + 1) % self.mini_batch_size) == 0:
        #         mini_batch_inputs.append({'batch_size': self.mini_batch_size,
        #                                   'task_instruction': self.cur_instruct,
        #                                   'samples': chain_input})
        #         chain_input = ''
        # if not (chain_input == ''):
        #     mini_batch_inputs.append({'batch_size': self.mini_batch_size,
        #                               'task_instruction': self.cur_instruct,
        #                               'samples': chain_input})
        # all_results = self.chain.batch_invoke(mini_batch_inputs, self.num_workers)
        # union_results = [element for sublist in all_results for element in sublist['results']]
        # for res in union_results:
        #     records.loc[res['id'], self.mode] = res['prediction']

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
