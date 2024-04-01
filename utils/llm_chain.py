from langchain.chains.openai_functions import (
    create_structured_output_runnable)
from utils.config import get_llm, load_prompt
from langchain_community.callbacks import get_openai_callback
import asyncio
from langchain.chains import LLMChain
import importlib
from pathlib import Path
from tqdm import trange, tqdm
import concurrent.futures
import logging


class DummyCallback:
    """
    A dummy callback for the LLM.
    This is a trick to handle an empty callback.
    """

    def __enter__(self):
        self.total_cost = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def get_dummy_callback():
    return DummyCallback()


class ChainWrapper:
    """
    A wrapper for a LLM chain
    """

    def __init__(self, llm_config, prompt_path: str, json_schema: dict = None, parser_func=None):
        """
        Initialize a new instance of the ChainWrapper class.
        :param llm_config: The config for the LLM
        :param prompt_path: A path to the prompt file (text file)
        :param json_schema: A dict for the json schema, to get a structured output for the LLM
        :param parser_func: A function to parse the output of the LLM
        """
        self.llm_config = llm_config
        self.llm = get_llm(llm_config)
        self.json_schema = json_schema
        self.parser_func = parser_func
        self.prompt = load_prompt(prompt_path)
        self.build_chain()
        self.accumulate_usage = 0
        if self.llm_config.type == 'OpenAI':
            self.callback = get_openai_callback
        else:
            self.callback = get_dummy_callback

    def invoke(self, chain_input: dict) -> dict:
        """
        Invoke the chain on a single input
        :param chain_input: The input for the chain
        :return: A dict with the defined json schema
        """
        with self.callback() as cb:
            try:
                result = self.chain.invoke(chain_input)
                if self.parser_func is not None:
                    result = self.parser_func(result)
            except Exception as e:
                if e.http_status == 401:
                    raise e
                else:
                    logging.error('Error in chain invoke: {}'.format(e.user_message))
                    result = None
            self.accumulate_usage += cb.total_cost
            return result

    async def retry_operation(self, tasks):
        """
        Retry an async operation
        :param tasks:
        :return:
        """
        delay = self.llm_config.async_params.retry_interval
        timeout = delay * self.llm_config.async_params.max_retries

        start_time = asyncio.get_event_loop().time()
        end_time = start_time + timeout
        results = []
        while True:
            remaining_time = end_time - asyncio.get_event_loop().time()
            if remaining_time <= 0:
                print("Timeout reached. Operation incomplete.")
                break

            done, pending = await asyncio.wait(tasks, timeout=delay)
            results += list(done)

            if len(done) == len(tasks):
                print("All tasks completed successfully.")
                break

            if not pending:
                print("No pending tasks. Operation incomplete.")
                break

            tasks = list(pending)  # Retry with the pending tasks
        return results

    async def async_batch_invoke(self, inputs: list[dict]) -> list[dict]:
        """
        Invoke the chain on a batch of inputs in async mode
        :param inputs: A batch of inputs
        :return: A list of dicts with the defined json schema
        """
        with self.callback() as cb:
            tasks = [self.chain.ainvoke(chain_input) for chain_input in inputs]
            all_res = await self.retry_operation(tasks)
            self.accumulate_usage += cb.total_cost
            if self.parser_func is not None:
                return [self.parser_func(t.result()) for t in list(all_res)]
            return [t.result() for t in list(all_res)]

    def batch_invoke(self, inputs: list[dict], num_workers: int):
        """
        Invoke the chain on a batch of inputs either async or not
        :param inputs: The list of all inputs
        :param num_workers: The number of workers
        :return: A list of results
        """

        def sample_generator():
            for sample in inputs:
                yield sample

        def process_sample_with_progress(sample):
            result = self.invoke(sample)
            pbar.update(1)  # Update the progress bar
            return result

        if not ('async_params' in self.llm_config.keys()):  # non async mode, use regular workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                with tqdm(total=len(inputs), desc="Processing samples") as pbar:
                    all_results = list(executor.map(process_sample_with_progress, sample_generator()))
        else:
            all_results = []
            for i in trange(0, len(inputs), num_workers, desc='Predicting'):
                results = asyncio.run(self.async_batch_invoke(inputs[i:i + num_workers]))
                all_results += results
        all_results = [res for res in all_results if res is not None]
        return all_results

    def build_chain(self):
        """
        Build the chain according to the LLM type
        """
        if (self.llm_config.type == 'OpenAI' or self.llm_config.type == 'Azure') and self.json_schema is not None:
            self.chain = create_structured_output_runnable(self.json_schema, self.llm, self.prompt)
        else:
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt)


def get_chain_metadata(prompt_fn: Path, retrieve_module: bool = False) -> dict:
    """
    Get the metadata of the chain
    :param prompt_fn: The path to the prompt file
    :param retrieve_module: If True, retrieve the module
    :return: A dict with the metadata
    """
    prompt_directory = str(prompt_fn.parent)
    prompt_name = str(prompt_fn.stem)
    try:
        spec = importlib.util.spec_from_file_location('output_schemes', prompt_directory + '/output_schemes.py')
        schema_parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schema_parser)
    except ImportError as e:
        print(f"Error loading module {prompt_directory + '/output_schemes'}: {e}")

    if hasattr(schema_parser, '{}_schema'.format(prompt_name)):
        json_schema = getattr(schema_parser, '{}_schema'.format(prompt_name))
    else:
        json_schema = None
    if hasattr(schema_parser, '{}_parser'.format(prompt_name)):
        parser_func = getattr(schema_parser, '{}_parser'.format(prompt_name))
    else:
        parser_func = None
    result = {'json_schema': json_schema, 'parser_func': parser_func}
    if retrieve_module:
        result['module'] = schema_parser
    return result


class MetaChain:
    """
    A wrapper for the meta-prompts chain
    """

    def __init__(self, config):
        """
        Initialize a new instance of the MetaChain class. Loading all the meta-prompts
        :param config: An EasyDict configuration
        """
        self.config = config
        self.initial_chain = self.load_chain('initial')
        self.step_prompt_chain = self.load_chain('step_prompt')
        self.step_samples = self.load_chain('step_samples')
        self.error_analysis = self.load_chain('error_analysis')

    def load_chain(self, chain_name: str) -> ChainWrapper:
        """
        Load a chain according to the chain name
        :param chain_name: The name of the chain
        """
        metadata = get_chain_metadata(self.config.meta_prompts.folder / '{}.prompt'.format(chain_name))
        return ChainWrapper(self.config.llm, self.config.meta_prompts.folder / '{}.prompt'.format(chain_name),
                            metadata['json_schema'], metadata['parser_func'])

    def calc_usage(self) -> float:
        """
        Calculate the usage of all the meta-prompts
        :return: The total usage value
        """
        return self.initial_chain.accumulate_usage + self.step_prompt_chain.accumulate_usage \
               + self.step_samples.accumulate_usage
