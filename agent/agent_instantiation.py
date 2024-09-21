from utils.llm_chain import ChainWrapper
from utils.config import get_llm
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict
from agent.agent_utils import build_agent
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
        if self.local_scope is None:
            self.local_scope = {}
        # exec(self.function_implementation, globals(), local_scope)
        self.local_scope.update(local_scope)

    def instantiate_node(self, tools):
        """
        Instantiate the node
        """
        llm = get_llm(self.function_metadata['llm'])
        self.function_implementation = build_agent(llm, tools, self.function_metadata, is_debug=True)

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
        self.llm_config = config_params.llm
        self.llm = get_llm(config_params.llm)
        self.chain_yaml_extraction = ChainWrapper(config_params['llm'],
                                                  'prompts/meta_prompts_agent/extract_yaml.prompt',
                                                  None, None)
        self.tools = tools

    def build_agent_function(self, agent_info):
        """
        wrap the agent in a function
        :param agent_info: The metadata of the agent
        """
        agent = build_agent(self.llm, self.tools, agent_info, is_debug=False)

        def new_function(**kwargs):
            # Pre-processing: Log the call
            input_str = ''
            for t, v in kwargs.items():
                input_str += '{}: {}\n'.format(t, v)
            input_str = input_str[:-1]
            results = agent.invoke({'input': input_str})
            final_res = []
            for var in agent_info['outputs']:
                if var.name not in results:
                    final_res.append('Variable {} not found in the results'.format(var.name))
                else:
                    final_res.append(results[var.name])
            if len(final_res) == 1:
                return final_res[0]
            return final_res

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
        agent_node.function_metadata['llm'] = dict(self.llm_config)
        return agent_node

def get_var_schema(var_metadata: list[Variable], style='yaml'):
    """
    Rephrase the schema and providing a string in the given provided style
    :param var_metadata: The metadata of the variables
    :param style: The style of the output (yaml, json, plain)
    """
    if style == 'json':
        output_schema = '{'
        for var in var_metadata:
            output_schema += '\n'
            output_schema += '{}: {{type: {}, description: {}}},'.format(var.name, var.type, var.description)
        output_schema += '\n}\n'
    elif style == 'yaml':
        output_schema = ''
        for var in var_metadata:
            output_schema += '{}: {} #{}\n'.format(var.name, var.type, var.description)
    else:
        output_schema = ''
        for var in var_metadata:
            output_schema += '{}: {} \n'.format(var.name, var.description)
    return output_schema[:-1]