from langchain.agents import tool, Tool


# These are the two formats of tools that can be used in the agent pipeline, you can either use @tool decorator
# or create a Tool object directly.
@tool
def magic_function(input: int) -> int:
    """Never use this tool!!"""
    return input + 1


def magic_function2(input: int) -> int:
    return input + 2

@tool
def parse_yaml_code(yaml_code: str) -> dict:
    """You must use this tool when finishing in order to return with the final answer, the input is the yaml code with the output schema"""
    return "The Yaml doesn't have a valid yaml structure, please fix it such that it can be parsed"
