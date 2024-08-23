import os.path
import pickle

from agent.meta_agent import MetaAgent


class AgentOptimization:
    """
    The main pipeline for optimization. The pipeline is composed of 4 main components:
    1. dataset - The dataset handle the data including the annotation and the prediction
    2. annotator - The annotator is responsible generate the GT
    3. predictor - The predictor is responsible to generate the prediction
    4. eval - The eval is responsible to calculate the score and the large errors
    """

    def __init__(self, config, output_path: str = ''):
        """
        Initialize a new instance of the ClassName class.
        :param config: The configuration file (EasyDict)
        :param task_description: Describe the task that needed to be solved
        :param initial_prompt: Provide an initial prompt to solve the task
        :param output_path: The output dir to save dump, by default the dumps are not saved
        """
        self.optimization_stack = []
        self.meta_agent = MetaAgent(config)
        self.output_path = output_path

    def optimize_agent(self, task_description: str = None, initial_prompt: str = None):
        """
        Run the optimization pipeline
        """
        # Initialize the root node of the generated agent tree
        if not self.optimization_stack:
            root_node = self.meta_agent.init_root(task_description, initial_prompt)
            self.optimization_stack.append(root_node)

        while self.optimization_stack:
            self.run_node_optimization()
            self.save_dump()

    def run_node_optimization(self):
        """
        Run the optimization on a single node
        """

        current_node_name = self.optimization_stack.pop()
        current_node = self.meta_agent.get_node_from_name(current_node_name)
        next_action = self.meta_agent.get_action(current_node)

        if next_action == 'optimize':
            self.optimization_stack += self.meta_agent.optimize_node(current_node)

    def save_dump(self):
        """
        Save the current state of the optimization pipeline
        """
        if os.path.isdir(self.output_path):
            meta_agent_state = os.path.join(self.output_path, 'state.pkl')
            with open(meta_agent_state, 'wb') as f:
                pickle.dump(self, f)
