import yaml
from langchain.chains import LLMChain
from utils.llm_chain import ChainWrapper
from utils.config import get_llm
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict
from langchain_core.pydantic_v1 import BaseModel, Field


class Variable(BaseModel):
    type: str = Field(description="variable type")
    name: str = Field(description="variable name")
    description: str = Field(description="variable documentation")


class NodeType(Enum):
    INTERNAL = "internal"
    LEAF = "leaf"


@dataclass
class AgentNode:
    """
    A class that represent a node in the generated agent graph
    """
    node_type: NodeType
    quality: Dict = field(default_factory=lambda: {'updated': False, 'score': -1})  # The quality metadata of the node
    function_metadata: Dict = None  # The function metadata of the node
    function_implementation: str = None  # The function implementation call
    local_scope: Dict = None  # The local scope of the node

    def update_local_scope(self, local_scope: dict = {}):
        """
        Update the local scope of the node
        """
        # exec(self.function_implementation, globals(), local_scope)
        self.local_scope = local_scope

    def instantiate_node(self, input_globals):
        """
        Instantiate the node
        """
        exec(self.function_implementation, input_globals, self.local_scope)

    def __getstate__(self):
        # Return a dictionary of picklable attributes
        state = self.__dict__.copy()
        # Remove the non-picklable attribute
        del state['local_scope']
        return state


class FunctionBuilder:
    """
    Building a functions (either agents or flows) from metadata
    """

    def __init__(self, config_params, tools):
        """
        Initialize the function builder
        :param config_params: The configuration parameters
        :param tools: The available tools for the agent
        """
        self.llm = get_llm(config_params.llm)
        self.chain_yaml_extraction = ChainWrapper(config_params['llm'],
                                                  'prompts/meta_prompts_agent/extract_yaml.prompt',
                                                  None, None)
        self.tools = tools

    @staticmethod
    def parse_yaml(response: dict):
        # Parse the YAML file from the model response
        pattern = r"##YAML file##\s*(.*?)\s*##end file##"
        # Search for the pattern in the text
        match = re.search(pattern, response['text'], re.DOTALL | re.MULTILINE)
        content = match.group(1)
        return yaml.safe_load(content)

    @staticmethod
    def parse(text: str) -> dict:
        return json.loads(text)

    @staticmethod
    def extract_response(text: dict) -> dict:
        return {'response': text['text']}

    def build_agent(self, agent_info):
        """
        Build an agent from metadata
        :param agent_info: The metadata of the agent
        """
        cur_tools = [t for t in self.tools if t.name in agent_info['tools']]
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

            agent = create_tool_calling_agent(self.llm, cur_tools, prompt)
            # agent = agent | StrOutputParser() | parse
            agent_executor = AgentExecutor(
                agent=agent, tools=cur_tools, verbose=True
            )
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
            agent = LLMChain(llm=self.llm, prompt=prompt)
            #TODO: Remove the chain_yaml_extraction, the results should directly the YAML
            agent_executor = agent | FunctionBuilder.extract_response \
                             | self.chain_yaml_extraction.chain | FunctionBuilder.parse_yaml

        return agent_executor

    def build_agent_function(self, agent_info):
        """
        wrap the agent in a function
        :param agent_info: The metadata of the agent
        """
        agent = self.build_agent(agent_info)

        def new_function(**kwargs):
            # Pre-processing: Log the call
            input_str = ''
            for t, v in kwargs.items():
                input_str += '{}: {}\n'.format(t, v)
            input_str = input_str[:-1]
            results = agent.invoke({'input': input_str})
            return results

        return new_function

    def build_function(self, function_info):
        """
        Build a function from metadata (either flow or agent)
        :param function_info: The metadata of the function
        """
        local_scope = {}
        if function_info['type'] == 'agent':
            agent_function = self.build_agent_function(function_info)
            function_implementation = '{} = agent_function'.format(function_info['name'])
            agent_node = AgentNode(NodeType.LEAF, function_metadata=function_info,
                                   function_implementation=function_implementation)
            local_scope['agent_function'] = agent_function
        else:
            agent_node = AgentNode(NodeType.INTERNAL, function_metadata=function_info,
                                   function_implementation=function_info['code'])
        agent_node.update_local_scope(local_scope)
        return agent_node
