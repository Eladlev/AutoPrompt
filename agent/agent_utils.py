import importlib
import yaml
from langchain.chains import LLMChain
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from tqdm import trange, tqdm
import concurrent.futures
import logging
from utils.llm_chain import dict_to_prompt_text
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.format_scratchpad.tools import (
    format_to_tool_messages,
)
from agent.agent_tool_call import ToolsAgentOutputParser, extract_yaml_content


def parse(response: dict):
    result = extract_yaml_content(response['output'])
    return result


def load_tools(tools_path: str):
    """
    Load the agent tools from the function file
    """
    tools = []
    try:
        spec = importlib.util.spec_from_file_location('agent_tools', tools_path)
        schema_parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schema_parser)
    except ImportError as e:
        raise ImportError(f"Error loading module {tools_path}: {e}")
    # <class 'langchain_core.tools.StructuredTool'>
    for attribute in dir(schema_parser):
        # Skip special attributes
        if not attribute.startswith("__"):
            value = getattr(schema_parser, attribute)
            attr_type = str(type(value))
            # This is hardcoded for now, should be careful when updating langchain version
            if "<class 'langchain_core.tools" in attr_type:
                tools.append(value)
    return tools

def get_tools_description(tools: str or list):
    """
    Get the tools information
    """
    if isinstance(tools, str):
        tools = load_tools(tools)
    tools_dict = {tool.name: tool.description for tool in tools}
    if len(tools_dict) == 0:
        return "The agent doesn't have any available tool!!", tools_dict
    return dict_to_prompt_text({tool.name: tool.description for tool in tools}), tools_dict


def parse_yaml(response: dict):
    # Parse the YAML file from the model response
    pattern = r"##YAML file##\s*(.*?)\s*##end file##"
    # Search for the pattern in the text
    match = re.search(pattern, response['text'], re.DOTALL | re.MULTILINE)
    content = match.group(1)
    return yaml.safe_load(content)



def extract_response(text: dict) -> dict:
    return {'response': text['text']}


def build_agent(llm, tools, agent_info, intermediate_steps=False, is_debug=True):
    """
    Build an agent from metadata
    :param agent_info: The metadata of the agent
    :param llm: The language model
    :param tools: The available tools for the agent
    :param intermediate_steps: If True, return intermediate steps
    :param is_debug: If True, print intermediate steps and remove the yaml parsing
    """
    if 'tools' not in agent_info.keys():
        cur_tools = tools
    else:
        cur_tools = [t for t in tools if t.name in agent_info['tools']]
    if 'tools_metadata' in agent_info.keys():
        for tool in cur_tools:
            if tool.name in agent_info['tools_metadata']:
                tool.description = agent_info['tools_metadata'][tool.name]
    if len(cur_tools) > 0:

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    agent_info['prompt'],
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        llm_with_tools = llm.bind_tools(cur_tools)

        agent = (
                RunnablePassthrough.assign(
                    agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"])
                )
                | prompt
                | llm_with_tools
                | ToolsAgentOutputParser()
        )

        agent_executor = AgentExecutor(
            agent=agent, tools=cur_tools, verbose=True, return_intermediate_steps=intermediate_steps)

        if not is_debug:
            agent_executor = agent_executor | parse
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    agent_info['prompt'],
                ),
                ("human", "{input}"),
            ]
        )
        agent_executor = LLMChain(llm=llm, prompt=prompt)
        # TODO: Remove the chain_yaml_extraction, the results should directly the YAML
        # agent_executor = agent | extract_response \
        #                  | chain_yaml_extraction.chain | parse_yaml
        # agent_executor = AgentExecutor(
        #     agent=agent, tools=cur_tools, verbose=True)  # return_intermediate_steps=intermediate_steps

    return agent_executor


def batch_invoke(agent: AgentExecutor, inputs: list[dict], num_workers: int, callback) -> list[dict]:
    """
    Invoke the chain on a batch of inputs either async or not
    :param agent: The agent
    :param inputs: The list of all inputs
    :param num_workers: The number of workers
    :param callback: Langchain callback
    :return: A list of results
    """

    def sample_generator():
        for i, sample in enumerate(inputs):
            yield i, sample

    def process_sample_with_progress(sample):
        i, sample = sample
        with callback() as cb:
            try:
                result = agent.invoke(sample)
            except Exception as e:
                logging.error('Error in chain invoke: {}'.format(e))
                result = None
            accumulate_usage = cb.total_cost
        pbar.update(1)  # Update the progress bar
        return {'index': i, 'result': result, 'usage': accumulate_usage}

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(inputs), desc="Processing samples") as pbar:
            all_results = list(executor.map(process_sample_with_progress, sample_generator()))

    all_results = [res for res in all_results if res is not None]
    return all_results
