
#TODO: Remove form V2 branch

from prompts import RETRIEVAL_SCORE_PROMPT, PLANNING_SCORE_PROMPT, ERROR_ANALYSIS_PROMPT, AGENT_PROMPT_OPTIMIZATION, ASSIGNED_TOOL_PROMPT_OPTIMIZATION

class AgentOptimizer:

    """
    A class that responsible for optimizing tool agent prompt and tool descriptions
    """

    def __init__(self, chain, tools):
        self.tools = tools
        self.chain = chain
        self.threshold = 0.5 # threshold for confidence score for tool description optimization
    
    def run_optimization(self, agent_info, task, action_plan):
        self.agent_info = agent_info
        self.task = task
        self.action_plan = action_plan
        self.tools_assigned = self.agent_info['tools']

        scores = self.get_scores()

        ## Agent prompt refinement
        agent_scores = {}
        agent_scores['planning_score'] = scores['planning_score']
        agent_scores['retrieval_score'] = sum(scores['retrieval_score']) / len(scores['retrieval_score']) # Mean of confidence scores of all the assigned tools
        refined_agent_prompt = self.optimize_agent_prompt(agent_scores)

        ## Tools description refinement
        tool_scores = {}
        for i in range(len(self.tools_assigned)):
            tool_scores[self.tools_assigned[i].name] = scores['retrieval_score'][i]

        refined_tool_descriptions = self.optimize_tools(tool_scores)

        return refined_agent_prompt, refined_tool_descriptions

    def get_scores(self):

        ## Static Scores
        tool_descriptions = [tool.description for tool in self.tools_assigned]
        retrieval_score = eval(self.chain.invoke(RETRIEVAL_SCORE_PROMPT.format(task = self.task, plan = self.action_plan, tool_descriptions = tool_descriptions)))
        planning_score = eval(self.chain.invoke(PLANNING_SCORE_PROMPT.format(task = self.task, plan = self.action_plan)))

        ## Generated Metrics needs to be added

        return {'retrieval_score' : retrieval_score, 'planning_score': planning_score}

    def optimize_agent_prompt(self, agent_scores):
        error_analysis = self.chain.invoke(ERROR_ANALYSIS_PROMPT.format(task = self.task, agent_prompt = self.agent_info['prompt'], plan = self.action_plan, retrieval_score = agent_scores['retrieval_score'], planning_score = agent_scores['planning_score']))
        refined_agent_prompt = self.chain.invoke(AGENT_PROMPT_OPTIMIZATION.format(task = self.task, agent_prompt = self.agent_info['prompt'], plan = self.action_plan, retrieval_score = agent_scores['retrieval_score'], planning_score = agent_scores['planning_score'], error_analysis = error_analysis))
        return refined_agent_prompt
 
    def optimize_tools(self, tool_scores):
        for tool in self.tools:
            if tool in self.tools_assigned and tool_scores[tool.name] < self.threshold:
                self.optimize_tool_prompt(tool, True, tool_scores[tool.name])
            else:
                self.optimize_tool_prompt(tool, False, None)

    def optimize_tool_prompt(self, tool, is_assigned, score):
        if is_assigned:
            refined_tool_description = self.chain.invoke(ASSIGNED_TOOL_PROMPT_OPTIMIZATION.format(task = self.task, description = tool.description, score = score))
        else:
            refined_tool_description = None # Yet to figure out how to implement
        
        return refined_tool_description
