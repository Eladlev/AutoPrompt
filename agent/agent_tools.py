from langchain.agents import tool, Tool

import yaml

from langchain.agents import tool, Tool



# These are the two formats of tools that can be used in the agent pipeline, you can either use @tool decorator
# or create a Tool object directly.
@tool
def magic_function(input: int) -> dict:
    """Use this tool meaningfully and only if its results are relevant to the final response. Ensure the input for this tool is a random integer"""
    return {'result': input + 1}


@tool
def parse_yaml_code(yaml_code: str) -> dict:
    """You must use this tool before sending the final output, the input is the yaml code with the output schema. The result is the final output!"""
    return "The Yaml doesn't have a valid yaml structure, please fix it such that it can be parsed. Remember that if you have a value that is a string, you should wrap it in quotes."
