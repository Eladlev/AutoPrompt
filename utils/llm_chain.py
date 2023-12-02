from langchain.chains.openai_functions import (
    create_structured_output_runnable)
from utils.config import get_llm, load_prompt
import utils.output_scehmes as json_schemas
from langchain.callbacks import get_openai_callback
import asyncio


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

    def __init__(self, llm_config, prompt_path: str, json_schema: dict):
        """
        Initialize a new instance of the ChainWrapper class.
        :param llm_config: The config for the LLM
        :param prompt_path: A path to the prompt file (text file)
        :param json_schema: A dict for the json schema, to get a structured output for the LLM
        """
        self.llm_config = llm_config
        self.llm = get_llm(llm_config)
        self.json_schema = json_schema
        appendix = self.get_appendix()
        self.prompt = load_prompt(prompt_path, appendix)
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
            result = self.chain.invoke(chain_input)
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
            return [t.result() for t in list(all_res)]

    def get_appendix(self):
        """
        Get the appendix to generate the correct json_schema
        :return: Either None or a string
        """
        if self.llm_config.type == 'HuggingFaceHub':
            return f"answer in the following json schema: {self.json_schema} :"
        else:
            return None

    def build_chain(self):
        """
        Build the chain according to the LLM type
        """
        if self.llm_config.type == 'OpenAI':
            self.chain = create_structured_output_runnable(self.json_schema, self.llm, self.prompt)
        elif self.llm.model.type == 'HuggingFaceHub':
            # TODO: add from here https://github.com/noamgat/lm-format-enforcer/blob/fccfee7a9dd23ef2c0a6d9aa1cdd084a1b922383/samples/colab_llama2_enforcer.ipynb#L1010
            raise NotImplementedError("HuggingFaceHub not implemented")
        else:
            raise NotImplementedError("LLM not implemented")


class MetaChain:
    """
    A wrapper for the meta-prompts chain
    """

    def __init__(self, config):
        """
        Initialize a new instance of the MetaChain class. Loading all the meta-prompts
        :param config: An EasyDict configuration
        """
        self.initial_chain = ChainWrapper(config.llm, config.meta_prompts.folder / 'initial.prompt',
                                          json_schemas.sample_generation_schema)
        self.step_prompt_chain = ChainWrapper(config.llm, config.meta_prompts.folder / 'step_prompt.prompt',
                                              json_schemas.step_prompt_schema)
        self.step_samples = ChainWrapper(config.llm, config.meta_prompts.folder / 'step_samples.prompt',
                                         json_schemas.sample_generation_schema)

    def calc_usage(self) -> float:
        """
        Calculate the usage of all the meta-prompts
        :return: The total usage value
        """
        return self.initial_chain.accumulate_usage + self.step_prompt_chain.accumulate_usage \
                                                   + self.step_samples.accumulate_usage
