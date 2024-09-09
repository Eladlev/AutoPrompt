from utils.llm_chain import dict_to_prompt_text
from agent.agent_instantiation import AgentNode, Variable, NodeType
from utils.llm_chain import MetaChain
from agent.agent_instantiation import FunctionBuilder, get_var_schema
from collections import deque
from agent.agent_utils import load_tools
from agent.node_optimization import run_agent_optimization, run_flow_optimization


class MetaAgent:
    """
    The MetaAgent class is responsible to optimize any given agent
    """

    def __init__(self, config, output_path: str = ''):
        """
        Initialize a new instance of the MetaAgent class.
        :param config: The configuration file (EasyDict)
        """
        self.config = config
        self.output_path = output_path
        self.meta_chain = MetaChain(config)
        self.code_tree = None
        self.code_root = None
        self.tools = []
        self.tools = load_tools(self.config.agent_config.tools_path)
        self.tools_metadata = {t.name: t.description for t in self.tools}
        self.function_builder = FunctionBuilder(config, self.tools)

    @staticmethod
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

    @staticmethod
    def extract_tool_str(tools_metadata: dict):
        """
        Extract the tools metadata to a string
        :param tools_metadata: The metadata of the tools
        """
        return '\n ----- \n'.join(['{}: {}'.format(t, v) for t, v in tools_metadata.items()])

    def get_initial_system_prompt(self, task_description: str, input_variables: list[Variable],
                                  output_variables: list[Variable], cur_tools_metadata: dict) -> str:
        """
        Get the initial system prompt
        :param task_description: The agent description
        :param input_variables: The input variables
        :param output_variables: The output variables
        :param cur_tools_metadata: The metadata of the tools
        """
        meta_data_str = get_var_schema(output_variables, 'yaml')
        input_str = get_var_schema(input_variables, 'list')
        tools_str = MetaAgent.extract_tool_str(cur_tools_metadata)
        initial_prompt = self.meta_chain.chain.build_agent_init.invoke({
            'task_description': task_description,
            'input': input_str,
            'tools': tools_str,
            'yaml_schema': meta_data_str
        })
        return initial_prompt['prompt']

    def make_tree_code_runnable(self, global_scope: dict = {}):
        """
        Make the code tree runnable
        :param global_scope: The global scope to run the code tree
        """

        def bfs_collect_nodes(code_tree):
            if not code_tree:
                return []

            nodes_list = []
            queue = deque(['root'])

            while queue:
                current_node = queue.popleft()
                nodes_list.append(current_node)
                for child in code_tree[current_node]['children']:
                    queue.append(child)
            return nodes_list

        tree_nodes_list = bfs_collect_nodes(self.code_tree)[::-1]
        local_scope = {}
        for node_name in tree_nodes_list:
            cur_node = self.code_tree[node_name]['node']
            cur_node.instantiate_node(global_scope)
            local_scope[cur_node.function_metadata['name']] = cur_node.local_scope[cur_node.function_metadata['name']]
            global_scope.update(local_scope)
        return global_scope  # usage example:  exec('root(input="What is the return policy")', global_scope)

    def apply_agent_optimization(self, node: AgentNode):
        """
        Apply the agent optimization (initial optimization)
        :param node: The node to optimize
        """
        new_prompt_info = run_agent_optimization(node, self.output_path, self.config,
                                                 [tool for tool in self.tools if
                                                  tool.name in node.function_metadata['tools']])
        new_agent_function = self.function_builder.build_agent_function(node.function_metadata)
        node.update_local_scope({'agent_function': new_agent_function})
        node.quality = {'updated': True, 'score': new_prompt_info['score'],
                        'analysis': new_prompt_info['analysis'],
                        'score_info': dict_to_prompt_text(new_prompt_info['score_info']),
                        'metrics_info': new_prompt_info['metrics_info']}
        return node.function_metadata['name']

    def apply_flow_optimization(self, node: AgentNode):
        """
        Apply the flow optimization
        Currently it is very basic optimization- If there is any bug, it tries to fix the flow by updating the function
        However it use the same sub-component and not try to replace them
        :param node: The node to optimize
        """
        # Apply the meta-chain to get the flow optimization
        local_scope = {}
        try:
            local_scope = self.make_tree_code_runnable(local_scope)
            need_optimization = run_flow_optimization(node, self.output_path, local_scope, self.config,
                                                      [tool for tool in self.tools if
                                                       tool.name in node.function_metadata['tools']])
            analysis = node.quality['analysis']
        except Exception as e:
            analysis = f'The given code is not runnable, the compiler provide the following error: {str(e)}'
            need_optimization = True

        optimization_step_left = node.quality.get('optimization_step_left', 2)  # TODO: remove coded value
        if not need_optimization or optimization_step_left == 0:
            node.quality['updated'] = True
            return []

        res = self.meta_chain.chain['updating_flow'].invoke(
            {'task_description': node.function_metadata['function_description'],
             'code_block': node.function_implementation,
             'analysis': analysis})
        node.quality['optimization_step_left'] = optimization_step_left - 1
        node.function_implementation = res['code']
        return [node.function_metadata['name']]

    def apply_flow_decomposition(self, node: AgentNode):
        """
        Apply the flow decomposition optimization
        :param node: The node to optimize
        """
        # Apply the meta-chain to get the flow decomposition
        cur_tools_metadata = {t: self.tools_metadata[t] for t in node.function_metadata['tools'] if not t == 'parse_yaml_code'}
        tools_str = MetaAgent.extract_tool_str(cur_tools_metadata)
        flow_decomposition = self.meta_chain.chain.breaking_flow.invoke(
            {'function_name': node.function_metadata['name'],
             'inputs': get_var_schema(node.function_metadata['inputs']),
             'outputs': get_var_schema(node.function_metadata['outputs']),
             'tools': tools_str,
             'task_description': node.function_metadata['function_description'],
             'analysis': 'The agent score: {}, analysis: {}'.format(node.quality['score'], node.quality['analysis'])
             })

        node_results = []
        for child in flow_decomposition.sub_functions_list:
            # create the agent
            if 'parse_yaml_code' not in child.tools_list:
                child.tools_list.append('parse_yaml_code')
            initial_prompt = self.get_initial_system_prompt(child.function_description,
                                                            child.input_variables,
                                                            child.output_variables,
                                                            {t: self.tools_metadata[t] for t
                                                             in child.tools_list})

            cur_node = self.function_builder.build_function({'type': 'agent',
                                                             'name': child.function_name,
                                                             'tools': child.tools_list,
                                                             'function_description': child.function_description,
                                                             'prompt': initial_prompt,
                                                             'inputs': child.input_variables,
                                                             'outputs': child.output_variables})
            # Add the child to the code tree
            self.code_tree[child.function_name] = {'node': cur_node, 'children': []}
            self.code_tree[node.function_metadata['name']]['children'].append(child.function_name)
            node_results.append(cur_node.function_metadata['name'])

        node.node_type = NodeType.INTERNAL
        node.function_implementation = flow_decomposition.code_flow
        node.quality['updated'] = False
        node.update_local_scope()
        node_results.append(node.function_metadata['name'])
        return node_results[::-1]  # reverse the list to keep DFS

    def optimize_node(self, node: AgentNode):
        """
        Run the meta-prompts and get new prompt suggestion, estimated prompt score and a set of challenging samples
        for the new prompts
        """
        if node.node_type == NodeType.LEAF:
            if not node.quality['updated']:  # not previous set
                stack_update = [self.apply_agent_optimization(node)]
            else:
                stack_update = self.apply_flow_decomposition(node)

        elif node.node_type == NodeType.INTERNAL:
            if not node.quality['updated']:  # In this case we need to optimize the flow
                stack_update = self.apply_flow_optimization(node)
            else:  # TODO: add support in replacing components support
                stack_update = []
        else:
            raise Exception("Unknown node type")

        return stack_update

    def get_node_from_name(self, node_name: str):
        """
        Get the node from the code tree by the node name
        :param node_name: The name of the node
        """
        return self.code_tree[node_name]['node']

    def get_action(self, node: AgentNode):
        """
        Get the next action for a given node
        """
        # If the node needs to update the optimization, optimize it
        if node.quality['updated'] is False:
            return 'optimize'
        # If the node is a leaf,decide if to breakdown the agent to a flow
        if node.node_type == NodeType.LEAF:
            res = self.meta_chain.chain.action_decision_agent.invoke({
                'task_description': node.function_metadata['function_description'],
                'metrics_description': node.quality['metrics_info'],
                'task_analysis': 'The agent score: {}\nAnalysis: {}'.format(node.quality['score_info'],
                                                                            node.quality['analysis'])})
            return 'optimize' if res['decision'] else 'skip'

        # If the node is an internal node, decide if to rewrite the flow
        elif node.node_type == NodeType.INTERNAL:
            res = self.meta_chain.chain.action_decision_flow.invoke({
                'function_code': node.function_implementation,
                'task_analysis': 'The agent score: {}, analysis: {}'.format(node.quality['score'],
                                                                            node.quality['analysis'])})
            return 'optimize' if res['decision'] else 'skip'
        else:
            raise Exception("Unknown node type")

    def init_root(self, task_description, initial_prompt):
        """
        Initialize the root node of the generated agent tree
        """
        output_variable = Variable(name='result', type='str', description='The final response of the agent')
        input_variable = Variable(name='input', type='str', description='The input to the agent')

        if task_description == '':
            if initial_prompt == '':
                raise Exception("No task description or initial prompt was provided")
            # Get the task description from the initial prompt
            task_description = self.meta_chain.chain.initial.invoke({'system_prompt': initial_prompt})
            task_description = task_description['agent_description']

        # If the initial prompt is empty, get the initial prompt from the task description
        if initial_prompt == '':
            initial_prompt = self.get_initial_system_prompt(task_description, [input_variable], [output_variable],
                                                            self.tools_metadata)

        # Create the root node
        root_node = self.function_builder.build_function({'type': 'agent',
                                                          'name': 'root',
                                                          'tools': [t.name for t in self.tools],
                                                          'function_description': task_description,
                                                          'prompt': initial_prompt,
                                                          'inputs': [input_variable],
                                                          'outputs': [output_variable]})
        self.code_tree = {'root': {'node': root_node, 'children': []}}
        return root_node.function_metadata['name']

    def __getstate__(self):
        # Return a dictionary of picklable attributes
        state = self.__dict__.copy()
        # Remove the non-picklable attribute
        del state['meta_chain']
        del state['function_builder']
        del state['tools']
        return state

    def __setstate__(self, state):
        # Restore the object's state from the state dictionary
        self.__dict__.update(state)
        # Restore or reinitialize the non-picklable attribute
        self.meta_chain = MetaChain(self.config)
        self.tools = load_tools(self.config.agent_config.tools_path)
        self.tools_metadata = {t.name: t.description for t in self.tools}

        self.function_builder = FunctionBuilder(self.config, self.tools)
        # Go through the code tree and update the local scope
        for node_name, node in self.code_tree.items():
            local_scope = {}
            cur_node = node['node']
            if cur_node.node_type == NodeType.LEAF:
                local_scope['agent_function'] = self.function_builder.build_agent_function(cur_node.function_metadata)
            cur_node.update_local_scope(local_scope)
